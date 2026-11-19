package pigo

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
)

// Alignment limits. They keep pathological inputs (extremely small eye
// distances, absurd target sizes or scales) from overflowing coordinates or
// exhausting memory. They are deliberately generous: a face crop never needs
// to exceed them.
const (
	// MaxAlignScale is the largest accepted ratio between the target eye
	// distance and the distance between the source eye points.
	MaxAlignScale = 1e6
	// MaxAlignOutputPixels is the largest accepted output image area.
	MaxAlignOutputPixels = 1 << 26 // 67,108,864 pixels (256 MiB in NRGBA form)
)

// Alignment errors. They are returned unwrapped by sentinel identity (the
// returned error may wrap one of them with a short explanation, so use
// errors.Is to test for a specific cause).
var (
	ErrAlignNilImage           = errors.New("pigo: align face: nil source image")
	ErrAlignEmptyImage         = errors.New("pigo: align face: empty source image")
	ErrAlignInvalidFaceRect    = errors.New("pigo: align face: empty face rectangle")
	ErrAlignNonFinitePoint     = errors.New("pigo: align face: non-finite eye or anchor coordinate")
	ErrAlignInvalidEyeDistance = errors.New("pigo: align face: target eye distance must be positive and finite")
	ErrAlignCoincidentEyes     = errors.New("pigo: align face: eye points coincide")
	ErrAlignReversedEyes       = errors.New("pigo: align face: eye points are reversed")
	ErrAlignEyesOutsideFace    = errors.New("pigo: align face: eye point is outside the face rectangle")
	ErrAlignInvalidOutputSize  = errors.New("pigo: align face: output width and height must be positive")
	ErrAlignOutputTooLarge     = errors.New("pigo: align face: output image is too large")
	ErrAlignInvalidInterp      = errors.New("pigo: align face: unknown interpolation mode")
	ErrAlignScaleOverflow      = errors.New("pigo: align face: computed scale is non-finite or too large")
	ErrAlignSingularTransform  = errors.New("pigo: align face: affine transform is not invertible")
)

// Point is a location in image coordinates.
//
// Coordinates are continuous and pixel centers sit at integer values: the
// center of the pixel with index (x, y) is exactly (x, y), for any image
// bounds, including images whose Bounds().Min is not (0, 0). Pigo detectors
// return integer pixel indices, so a Puploc with Col == c and Row == r maps
// to Point{X: float64(c), Y: float64(r)}.
type Point struct {
	X float64
	Y float64
}

// Interpolation selects how the source image is sampled while generating the
// aligned image.
type Interpolation int

const (
	// Bilinear interpolates the four nearest source pixels. This is the zero
	// value and produces smooth results for scaled and rotated faces.
	Bilinear Interpolation = iota
	// NearestNeighbor picks the closest source pixel. It is faster and does
	// not blend colors, which is preferable for indexed or mask-like images.
	NearestNeighbor
)

// AffineMatrix is a two-dimensional affine transformation mapping points from
// the source image to the aligned output image:
//
//	x' = A*x + B*y + C
//	y' = D*x + E*y + F
//
// Use Invert to obtain the source-from-output mapping used for resampling.
type AffineMatrix struct {
	A, B, C float64
	D, E, F float64
}

// Apply returns the image of p under m.
func (m AffineMatrix) Apply(p Point) Point {
	return Point{
		X: m.A*p.X + m.B*p.Y + m.C,
		Y: m.D*p.X + m.E*p.Y + m.F,
	}
}

// Invert returns the inverse of m. It returns ErrAlignSingularTransform if m
// is degenerate or if the inverse cannot be represented with finite float64
// values.
func (m AffineMatrix) Invert() (AffineMatrix, error) {
	det := m.A*m.E - m.B*m.D
	if det == 0 || math.IsNaN(det) || math.IsInf(det, 0) {
		return AffineMatrix{}, ErrAlignSingularTransform
	}

	inv := AffineMatrix{
		A: m.E / det,
		B: -m.B / det,
		C: (m.B*m.F - m.E*m.C) / det,
		D: -m.D / det,
		E: m.A / det,
		F: (m.D*m.C - m.A*m.F) / det,
	}
	if !inv.isFinite() {
		return AffineMatrix{}, ErrAlignSingularTransform
	}
	return inv, nil
}

func (m AffineMatrix) isFinite() bool {
	for _, v := range [...]float64{m.A, m.B, m.C, m.D, m.E, m.F} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// AlignOptions configures AlignFace.
//
// LeftEye and RightEye are the eye centers in source image coordinates, with
// LeftEye being the point with the smaller X coordinate for an upright face.
// Face is the detected face rectangle, also in source coordinates; it is used
// to validate that the eye points are plausible. The output image is
// Width x Height pixels.
//
// The resulting transform rotates the face so that the eye line is horizontal
// and scales it so that the eye centers are TargetEyeDistance pixels apart.
// The midpoint between the eyes is mapped to EyeAnchor when EyeAnchor is not
// nil; otherwise it is mapped to the center of the output image.
//
// Pixels of the output that fall outside the source image are transparent
// when Background is nil and are filled with Background otherwise. Detections
// and the source image are never modified.
type AlignOptions struct {
	Face              image.Rectangle
	LeftEye           Point
	RightEye          Point
	TargetEyeDistance float64
	Width             int
	Height            int
	EyeAnchor         *Point
	Interpolation     Interpolation
	Background        color.Color
}

// AlignFace produces a rotation- and scale-normalized crop of src so that the
// eye points supplied in opts land at fixed, predictable coordinates in the
// output. It returns the aligned image and the affine matrix that maps source
// image coordinates to output image coordinates.
//
// The transform is computed as
//
//	output = scale * R(-angle) * (source - eyeMidpoint) + anchor
//
// where angle is the angle of the left-to-right eye vector. As a consequence,
// the two eye points map to anchor-(d/2, 0) and anchor+(d/2, 0), with d equal
// to opts.TargetEyeDistance.
//
// Points outside the source image are filled with opts.Background (or left
// transparent when Background is nil). Bilinear filtering treats such samples
// as background, so partially covered edge pixels blend smoothly into it.
//
// The function does not modify src or any detection results.
func AlignFace(src image.Image, opts AlignOptions) (*image.NRGBA, AffineMatrix, error) {
	if src == nil {
		return nil, AffineMatrix{}, ErrAlignNilImage
	}
	bounds := src.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return nil, AffineMatrix{}, ErrAlignEmptyImage
	}

	if opts.Width <= 0 || opts.Height <= 0 {
		return nil, AffineMatrix{}, fmt.Errorf("%w: %dx%d", ErrAlignInvalidOutputSize, opts.Width, opts.Height)
	}
	// Check each dimension first so the int64 multiplication below cannot
	// overflow even for absurd int values on 64-bit platforms.
	if opts.Width > MaxAlignOutputPixels || opts.Height > MaxAlignOutputPixels ||
		int64(opts.Width)*int64(opts.Height) > MaxAlignOutputPixels {
		return nil, AffineMatrix{}, fmt.Errorf("%w: %dx%d", ErrAlignOutputTooLarge, opts.Width, opts.Height)
	}
	if opts.Face.Empty() {
		return nil, AffineMatrix{}, ErrAlignInvalidFaceRect
	}
	switch opts.Interpolation {
	case Bilinear, NearestNeighbor:
	default:
		return nil, AffineMatrix{}, fmt.Errorf("%w: %d", ErrAlignInvalidInterp, opts.Interpolation)
	}

	// Non-finite inputs can otherwise propagate silently into the matrix or
	// into slice indices.
	if !isFinitePoint(opts.LeftEye) {
		return nil, AffineMatrix{}, fmt.Errorf("%w: left eye", ErrAlignNonFinitePoint)
	}
	if !isFinitePoint(opts.RightEye) {
		return nil, AffineMatrix{}, fmt.Errorf("%w: right eye", ErrAlignNonFinitePoint)
	}
	anchor := Point{
		X: float64(opts.Width-1) / 2,
		Y: float64(opts.Height-1) / 2,
	}
	if opts.EyeAnchor != nil {
		if !isFinitePoint(*opts.EyeAnchor) {
			return nil, AffineMatrix{}, fmt.Errorf("%w: eye anchor", ErrAlignNonFinitePoint)
		}
		anchor = *opts.EyeAnchor
	}
	if math.IsNaN(opts.TargetEyeDistance) || math.IsInf(opts.TargetEyeDistance, 0) || opts.TargetEyeDistance <= 0 {
		return nil, AffineMatrix{}, fmt.Errorf("%w: %v", ErrAlignInvalidEyeDistance, opts.TargetEyeDistance)
	}
	if !pointInRect(opts.LeftEye, opts.Face) {
		return nil, AffineMatrix{}, fmt.Errorf("%w: left eye (%g, %g)", ErrAlignEyesOutsideFace, opts.LeftEye.X, opts.LeftEye.Y)
	}
	if !pointInRect(opts.RightEye, opts.Face) {
		return nil, AffineMatrix{}, fmt.Errorf("%w: right eye (%g, %g)", ErrAlignEyesOutsideFace, opts.RightEye.X, opts.RightEye.Y)
	}

	dx := opts.RightEye.X - opts.LeftEye.X
	dy := opts.RightEye.Y - opts.LeftEye.Y
	dist := math.Hypot(dx, dy)
	if dist == 0 {
		return nil, AffineMatrix{}, ErrAlignCoincidentEyes
	}
	// A face whose "right" eye is left of its "left" eye would be rotated by
	// ~180 degrees and end up upside down. Reject it instead of silently
	// producing a useless crop. Exactly vertical eye pairs remain valid.
	if dx < 0 {
		return nil, AffineMatrix{}, ErrAlignReversedEyes
	}

	scale := opts.TargetEyeDistance / dist
	if scale <= 0 || math.IsNaN(scale) || math.IsInf(scale, 0) || scale > MaxAlignScale {
		return nil, AffineMatrix{}, fmt.Errorf("%w: %g", ErrAlignScaleOverflow, scale)
	}

	midX := (opts.LeftEye.X + opts.RightEye.X) / 2
	midY := (opts.LeftEye.Y + opts.RightEye.Y) / 2
	angle := math.Atan2(dy, dx)
	cos, sin := math.Cos(angle), math.Sin(angle)

	// Rotation by -angle followed by scaling and translation.
	m := AffineMatrix{
		A: scale * cos,
		B: scale * sin,
		C: anchor.X - scale*(cos*midX+sin*midY),
		D: -scale * sin,
		E: scale * cos,
		F: anchor.Y - scale*(-sin*midX+cos*midY),
	}
	if !m.isFinite() {
		return nil, AffineMatrix{}, fmt.Errorf("%w: non-finite transform", ErrAlignScaleOverflow)
	}
	inv, err := m.Invert()
	if err != nil {
		return nil, AffineMatrix{}, err
	}

	// Convert the source to unpremultiplied NRGBA on demand. Keeping the
	// background in both NRGBA (for nearest sampling) and premultiplied form
	// (for blending) avoids color fringes around transparent regions.
	bgNRGBA := color.NRGBA{}
	if opts.Background != nil {
		bgNRGBA = color.NRGBAModel.Convert(opts.Background).(color.NRGBA)
	}
	bgPremul := premultiply(bgNRGBA)

	dst := image.NewNRGBA(image.Rect(0, 0, opts.Width, opts.Height))
	sampler := sourceSampler{img: src, bounds: bounds}

	for y := 0; y < opts.Height; y++ {
		for x := 0; x < opts.Width; x++ {
			sx := inv.A*float64(x) + inv.B*float64(y) + inv.C
			sy := inv.D*float64(x) + inv.E*float64(y) + inv.F

			var c color.NRGBA
			if opts.Interpolation == NearestNeighbor {
				c = sampler.nearest(sx, sy, bgNRGBA)
			} else {
				c = sampler.bilinear(sx, sy, bgPremul)
			}
			o := dst.PixOffset(x, y)
			dst.Pix[o+0] = c.R
			dst.Pix[o+1] = c.G
			dst.Pix[o+2] = c.B
			dst.Pix[o+3] = c.A
		}
	}

	return dst, m, nil
}

func isFinitePoint(p Point) bool {
	return isFiniteCoord(p.X, p.Y)
}

// isFiniteCoord reports whether both coordinates are finite. NaN-infested
// coordinates must never reach float-to-int conversions, which are undefined
// for out-of-range values.
func isFiniteCoord(x, y float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0) &&
		!math.IsNaN(y) && !math.IsInf(y, 0)
}

// pointInRect reports whether p lies inside r. Eye coordinates are
// continuous, so the half-open pixel range [Min, Max) is used.
func pointInRect(p Point, r image.Rectangle) bool {
	return p.X >= float64(r.Min.X) && p.X < float64(r.Max.X) &&
		p.Y >= float64(r.Min.Y) && p.Y < float64(r.Max.Y)
}

// rgbaF is a premultiplied floating point color in the [0, 1] range.
type rgbaF struct {
	r, g, b, a float64
}

func premultiply(c color.NRGBA) rgbaF {
	a := float64(c.A) / 255
	return rgbaF{
		r: float64(c.R) / 255 * a,
		g: float64(c.G) / 255 * a,
		b: float64(c.B) / 255 * a,
		a: a,
	}
}

func unpremultiply(c rgbaF) color.NRGBA {
	a := clamp01(c.a)
	if a == 0 {
		return color.NRGBA{}
	}
	return color.NRGBA{
		R: uint8(math.Round(clamp01(c.r/a) * 255)),
		G: uint8(math.Round(clamp01(c.g/a) * 255)),
		B: uint8(math.Round(clamp01(c.b/a) * 255)),
		A: uint8(math.Round(a * 255)),
	}
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

type sourceSampler struct {
	img    image.Image
	bounds image.Rectangle
}

// nrgbaAt returns the source pixel at the integer image coordinates (x, y).
// The caller must have checked that the point is inside s.bounds.
func (s sourceSampler) nrgbaAt(x, y int) color.NRGBA {
	switch src := s.img.(type) {
	case *image.NRGBA:
		i := src.PixOffset(x, y)
		return color.NRGBA{R: src.Pix[i], G: src.Pix[i+1], B: src.Pix[i+2], A: src.Pix[i+3]}
	case *image.Gray:
		v := src.GrayAt(x, y).Y
		return color.NRGBA{R: v, G: v, B: v, A: 0xff}
	default:
		return color.NRGBAModel.Convert(s.img.At(x, y)).(color.NRGBA)
	}
}

// nearest samples the source at the continuous coordinate (x, y). Coordinates
// outside the valid pixel-center range produce the background color.
func (s sourceSampler) nearest(x, y float64, bg color.NRGBA) color.NRGBA {
	minX, maxX := float64(s.bounds.Min.X), float64(s.bounds.Max.X)
	minY, maxY := float64(s.bounds.Min.Y), float64(s.bounds.Max.Y)
	// Guard the float-to-int conversions below: out-of-range conversions are
	// implementation defined in Go and could produce negative indices.
	if !isFiniteCoord(x, y) || x < minX-0.5 || x >= maxX-0.5 || y < minY-0.5 || y >= maxY-0.5 {
		return bg
	}
	ix := int(math.Floor(x + 0.5))
	iy := int(math.Floor(y + 0.5))
	if ix < s.bounds.Min.X || ix >= s.bounds.Max.X || iy < s.bounds.Min.Y || iy >= s.bounds.Max.Y {
		return bg
	}
	return s.nrgbaAt(ix, iy)
}

// bilinear samples the source at the continuous coordinate (x, y), blending
// with the premultiplied background outside the image.
func (s sourceSampler) bilinear(x, y float64, bg rgbaF) color.NRGBA {
	minX, maxX := float64(s.bounds.Min.X), float64(s.bounds.Max.X)
	minY, maxY := float64(s.bounds.Min.Y), float64(s.bounds.Max.Y)
	if !isFiniteCoord(x, y) || x <= minX-1 || x >= maxX || y <= minY-1 || y >= maxY {
		return unpremultiply(bg)
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)

	c00 := s.premulOrBackground(x0, y0, bg)
	c10 := s.premulOrBackground(x0+1, y0, bg)
	c01 := s.premulOrBackground(x0, y0+1, bg)
	c11 := s.premulOrBackground(x0+1, y0+1, bg)

	w00 := (1 - fx) * (1 - fy)
	w10 := fx * (1 - fy)
	w01 := (1 - fx) * fy
	w11 := fx * fy

	return unpremultiply(rgbaF{
		r: c00.r*w00 + c10.r*w10 + c01.r*w01 + c11.r*w11,
		g: c00.g*w00 + c10.g*w10 + c01.g*w01 + c11.g*w11,
		b: c00.b*w00 + c10.b*w10 + c01.b*w01 + c11.b*w11,
		a: c00.a*w00 + c10.a*w10 + c01.a*w01 + c11.a*w11,
	})
}

func (s sourceSampler) premulOrBackground(x, y int, bg rgbaF) rgbaF {
	if x < s.bounds.Min.X || x >= s.bounds.Max.X || y < s.bounds.Min.Y || y >= s.bounds.Max.Y {
		return bg
	}
	return premultiply(s.nrgbaAt(x, y))
}
