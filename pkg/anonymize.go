package pigo

import (
	"errors"
	"image"
)

// AnonymizeMethod selects the anonymization algorithm applied to a masked region.
type AnonymizeMethod int

const (
	// Pixelate replaces each pixel block by the average color of its covered pixels.
	Pixelate AnonymizeMethod = iota
	// Blur applies a separable box blur with a configurable radius.
	Blur
)

const (
	// defaultBlockSize is the pixelation block edge length in pixels.
	defaultBlockSize = 10
	// defaultBlurRadius is the box blur radius in pixels (the kernel spans 2*r+1 pixels).
	defaultBlurRadius = 5
	// defaultMaxPixels bounds the number of pixels of a single image the API is willing
	// to process, keeping the temporary blur buffer (4 bytes per pixel) within a sane
	// memory envelope. A 50 megapixel image needs roughly 200 MB of scratch memory.
	defaultMaxPixels = 50_000_000
)

var (
	// ErrNilImage is returned when the input image or its pixel buffer is nil.
	ErrNilImage = errors.New("anonymize: input image is nil")
	// ErrEmptyImage is returned when the input image has zero width or height.
	ErrEmptyImage = errors.New("anonymize: input image has zero dimensions")
	// ErrFullyTransparent is returned when every pixel of the input image is fully
	// transparent: such an image carries no anonymizable visual information.
	ErrFullyTransparent = errors.New("anonymize: input image is fully transparent")
	// ErrImageTooLarge is returned when the image exceeds the configured MaxPixels limit.
	ErrImageTooLarge = errors.New("anonymize: image exceeds the maximum allowed pixel count")
	// ErrInvalidBlockSize is returned when a non-positive pixelation block size is configured.
	ErrInvalidBlockSize = errors.New("anonymize: pixelation block size must be greater than zero")
	// ErrInvalidBlurRadius is returned when a negative blur radius is configured.
	ErrInvalidBlurRadius = errors.New("anonymize: blur radius must not be negative")
	// ErrInvalidMethod is returned when the configured anonymization method is unknown.
	ErrInvalidMethod = errors.New("anonymize: unknown anonymization method")
)

// AnonymizeConfig holds the parameters of the face anonymization operation.
//
//   - Method: Pixelate (default) or Blur.
//   - Margin: extra pixels added on every side of each input rectangle. Negative
//     values shrink the rectangle; rectangles are always clipped to the image bounds.
//   - BlockSize: pixelation block edge length in pixels (only used by Pixelate).
//     A zero value falls back to the default of 10 pixels.
//   - BlurRadius: box blur radius in pixels (only used by Blur); the kernel spans
//     2*radius+1 pixels. A zero value falls back to the default radius of 5 pixels.
//   - MinFaceSize: detections whose Scale is smaller than this value are skipped.
//     Zero disables the minimum size check.
//   - MinConfidence: detections whose detection score (Detection.Q) is below this
//     threshold are skipped.
//   - Whitelist: rectangles whose pixels must never be modified, even when they
//     overlap an anonymization region.
//   - MaxPixels: upper bound on the processed image size (width * height), used to
//     cap scratch-memory allocation on very large images. A non-positive value
//     falls back to the built-in default (50 million pixels).
type AnonymizeConfig struct {
	Method        AnonymizeMethod
	Margin        int
	BlockSize     int
	BlurRadius    int
	MinFaceSize   int
	MinConfidence float32
	Whitelist     []image.Rectangle
	MaxPixels     int
}

// AnonymizeDetections anonymizes the face regions derived from Pigo detection results.
//
// Each Detection is interpreted as a square centered at (Col, Row) with the edge
// length Scale, matching the rectangle drawn by the library examples. The confidence
// threshold and minimum face size from the configuration filter the detections before
// the configured Margin is applied. The result is a new full-size *image.NRGBA image;
// the input image is never modified and pixels outside the anonymization regions,
// together with the full alpha channel, are preserved byte for byte.
func AnonymizeDetections(src *image.NRGBA, detections []Detection, cfg AnonymizeConfig) (*image.NRGBA, error) {
	rects := make([]image.Rectangle, 0, len(detections))
	for _, det := range detections {
		if det.Q < cfg.MinConfidence {
			continue
		}
		if cfg.MinFaceSize > 0 && det.Scale < cfg.MinFaceSize {
			continue
		}
		half := det.Scale / 2
		rects = append(rects, image.Rect(
			det.Col-half,
			det.Row-half,
			det.Col-half+det.Scale,
			det.Row-half+det.Scale,
		))
	}
	return Anonymize(src, rects, cfg)
}

// Anonymize anonymizes the provided rectangles on the source image and returns a new
// full-size *image.NRGBA image with the same bounds as the source.
//
// Rectangles may use negative coordinates or extend beyond the image bounds: they are
// clipped to the image and zero-area or fully out-of-bounds rectangles are skipped.
// Overlapping rectangles are merged first, so every covered pixel is transformed
// exactly once. Pixels covered by a whitelist rectangle and every alpha value are
// left untouched. The source image itself is never modified.
func Anonymize(src *image.NRGBA, rects []image.Rectangle, cfg AnonymizeConfig) (*image.NRGBA, error) {
	if src == nil || src.Pix == nil {
		return nil, ErrNilImage
	}
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return nil, ErrEmptyImage
	}

	maxPixels := cfg.MaxPixels
	if maxPixels <= 0 {
		maxPixels = defaultMaxPixels
	}
	switch cfg.Method {
	case Pixelate:
		if cfg.BlockSize < 0 {
			return nil, ErrInvalidBlockSize
		}
	case Blur:
		if cfg.BlurRadius < 0 {
			return nil, ErrInvalidBlurRadius
		}
	default:
		return nil, ErrInvalidMethod
	}

	if w*h > maxPixels {
		return nil, ErrImageTooLarge
	}
	if isFullyTransparent(src, bounds) {
		return nil, ErrFullyTransparent
	}

	// Clip and normalize the input rectangles, dropping anything with zero area.
	regions := make([]image.Rectangle, 0, len(rects))
	for _, r := range rects {
		if cfg.Margin != 0 {
			r = r.Inset(-cfg.Margin)
		}
		r = r.Intersect(bounds)
		if !r.Empty() {
			regions = append(regions, r)
		}
	}
	if len(regions) == 0 {
		return cloneNRGBA(src), nil
	}

	// Merge intersecting regions so a pixel is processed only once. Rectangles
	// that merely share an edge stay independent.
	regions = mergeRectangles(regions)

	// mask marks every pixel covered by an anonymization region.
	mask := make([]bool, w*h)
	for _, r := range regions {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			base := (y-bounds.Min.Y)*w + (r.Min.X - bounds.Min.X)
			for x := r.Min.X; x < r.Max.X; x++ {
				mask[base] = true
				base++
			}
		}
	}

	// The whitelist removes pixels from the mask before any transformation runs.
	for _, r := range cfg.Whitelist {
		r = r.Intersect(bounds)
		if r.Empty() {
			continue
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			base := (y-bounds.Min.Y)*w + (r.Min.X - bounds.Min.X)
			for x := r.Min.X; x < r.Max.X; x++ {
				mask[base] = false
				base++
			}
		}
	}

	dst := cloneNRGBA(src)

	switch cfg.Method {
	case Pixelate:
		blockSize := cfg.BlockSize
		if blockSize == 0 {
			blockSize = defaultBlockSize
		}
		pixelate(dst, regions, bounds, mask, blockSize)
	case Blur:
		radius := cfg.BlurRadius
		if radius == 0 {
			radius = defaultBlurRadius
		}
		blur(dst, bounds, mask, radius)
	}

	return dst, nil
}

// mergeRectangles unions all overlapping or edge-touching rectangles and returns the
// resulting disjoint set. The order of the returned rectangles is deterministic.
func mergeRectangles(rects []image.Rectangle) []image.Rectangle {
	merged := make([]image.Rectangle, 0, len(rects))
	used := make([]bool, len(rects))
	for i := 0; i < len(rects); i++ {
		if used[i] {
			continue
		}
		used[i] = true
		cur := rects[i]
		grew := true
		for grew {
			grew = false
			for j := i + 1; j < len(rects); j++ {
				if used[j] {
					continue
				}
				// Only genuinely intersecting rectangles are merged; rectangles
				// that merely share an edge stay independent regions.
				if cur.Overlaps(rects[j]) {
					cur = cur.Union(rects[j])
					used[j] = true
					grew = true
				}
			}
		}
		merged = append(merged, cur)
	}
	return merged
}

// pixelate replaces masked pixels by the per-block average of their RGB channels.
// Blocks are anchored at the top-left corner of each region; the alpha channel is
// never written.
func pixelate(img *image.NRGBA, regions []image.Rectangle, bounds image.Rectangle, mask []bool, blockSize int) {
	w := bounds.Dx()
	for _, r := range regions {
		for by := r.Min.Y; by < r.Max.Y; by += blockSize {
			blockBottom := by + blockSize
			if blockBottom > r.Max.Y {
				blockBottom = r.Max.Y
			}
			for bx := r.Min.X; bx < r.Max.X; bx += blockSize {
				blockRight := bx + blockSize
				if blockRight > r.Max.X {
					blockRight = r.Max.X
				}

				var sumR, sumG, sumB, count int
				for y := by; y < blockBottom; y++ {
					idx := (y-bounds.Min.Y)*w + (bx - bounds.Min.X)
					off := img.PixOffset(bx, y)
					for x := bx; x < blockRight; x++ {
						if mask[idx] {
							sumR += int(img.Pix[off])
							sumG += int(img.Pix[off+1])
							sumB += int(img.Pix[off+2])
							count++
						}
						idx++
						off += 4
					}
				}
				if count == 0 {
					continue
				}
				avgR := uint8(sumR / count)
				avgG := uint8(sumG / count)
				avgB := uint8(sumB / count)

				for y := by; y < blockBottom; y++ {
					idx := (y-bounds.Min.Y)*w + (bx - bounds.Min.X)
					off := img.PixOffset(bx, y)
					for x := bx; x < blockRight; x++ {
						if mask[idx] {
							img.Pix[off] = avgR
							img.Pix[off+1] = avgG
							img.Pix[off+2] = avgB
						}
						idx++
						off += 4
					}
				}
			}
		}
	}
}

// blur applies a separable box blur of the given radius to the masked pixels.
// Only masked RGB values are modified; each window is clipped to the image bounds
// and unmasked pixels keep their original values while being excluded from the
// averages. Every window is divided by the number of masked samples it actually
// contains, so whitelist holes do not bias the result. Alpha is always preserved.
func blur(img *image.NRGBA, bounds image.Rectangle, mask []bool, radius int) {
	w, h := bounds.Dx(), bounds.Dy()
	// tmp holds the horizontal-pass results for masked pixels only.
	tmp := make([][3]uint8, w*h)

	clampX := func(x int) int {
		if x < 0 {
			return 0
		}
		if x >= w {
			return w - 1
		}
		return x
	}
	clampY := func(y int) int {
		if y < 0 {
			return 0
		}
		if y >= h {
			return h - 1
		}
		return y
	}

	// Horizontal pass: average the masked pixels of the row window.
	for y := 0; y < h; y++ {
		rowBase := y * w
		for x := 0; x < w; x++ {
			idx := rowBase + x
			if !mask[idx] {
				continue
			}
			lo, hi := clampX(x-radius), clampX(x+radius)
			var sumR, sumG, sumB, count int
			off := img.PixOffset(bounds.Min.X+lo, bounds.Min.Y+y)
			for xx := lo; xx <= hi; xx++ {
				if mask[rowBase+xx] {
					sumR += int(img.Pix[off])
					sumG += int(img.Pix[off+1])
					sumB += int(img.Pix[off+2])
					count++
				}
				off += 4
			}
			tmp[idx] = [3]uint8{
				uint8(sumR / count),
				uint8(sumG / count),
				uint8(sumB / count),
			}
		}
	}

	// Vertical pass: average the horizontal results of the column window.
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			idx := y*w + x
			if !mask[idx] {
				continue
			}
			lo, hi := clampY(y-radius), clampY(y+radius)
			var sumR, sumG, sumB, count int
			for yy := lo; yy <= hi; yy++ {
				cur := yy*w + x
				if mask[cur] {
					sumR += int(tmp[cur][0])
					sumG += int(tmp[cur][1])
					sumB += int(tmp[cur][2])
					count++
				}
			}
			off := img.PixOffset(bounds.Min.X+x, bounds.Min.Y+y)
			img.Pix[off] = uint8(sumR / count)
			img.Pix[off+1] = uint8(sumG / count)
			img.Pix[off+2] = uint8(sumB / count)
		}
	}
}

// isFullyTransparent reports whether every pixel has a zero alpha value.
func isFullyTransparent(img *image.NRGBA, bounds image.Rectangle) bool {
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		off := img.PixOffset(bounds.Min.X, y)
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			if img.Pix[off+3] != 0 {
				return false
			}
			off += 4
		}
	}
	return true
}

// cloneNRGBA returns a deep copy of the source image preserving its bounds and stride.
func cloneNRGBA(src *image.NRGBA) *image.NRGBA {
	dst := image.NewNRGBA(src.Bounds())
	copy(dst.Pix, src.Pix)
	return dst
}
