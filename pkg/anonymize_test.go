package pigo_test

import (
	"image"
	"testing"

	pigo "github.com/esimov/pigo/pkg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// buildPatternImage creates a fixed w*h NRGBA image where every pixel has a
// deterministic value: R = 10*y+x, G = 100, B = 200 and a distinct alpha value.
// The fixed pixel layout is the reference used for the per-pixel assertions below.
func buildPatternImage(w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8(10*y + x)
			img.Pix[i+1] = 100
			img.Pix[i+2] = 200
			img.Pix[i+3] = uint8((7*x + 3*y + 1) % 256)
		}
	}
	return img
}

// buildCustomImage builds an NRGBA image from an explicit row-major R matrix;
// G and B stay constant and every pixel is fully opaque.
func buildCustomImage(r [][]uint8) *image.NRGBA {
	h := len(r)
	w := len(r[0])
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = r[y][x]
			img.Pix[i+1] = 100
			img.Pix[i+2] = 200
			img.Pix[i+3] = 255
		}
	}
	return img
}

func assertAlphaPreserved(t *testing.T, src, dst *image.NRGBA) {
	t.Helper()
	b := src.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; b.Max.X > x; x++ {
			assert.Equal(t, src.Pix[src.PixOffset(x, y)+3], dst.Pix[dst.PixOffset(x, y)+3],
				"alpha changed at (%d,%d)", x, y)
		}
	}
}

func assertOutsideUntouched(t *testing.T, src, dst *image.NRGBA, regions []image.Rectangle) {
	t.Helper()
	b := src.Bounds()
	masked := func(x, y int) bool {
		for _, r := range regions {
			if x >= r.Min.X && x < r.Max.X && y >= r.Min.Y && y < r.Max.Y {
				return true
			}
		}
		return false
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; b.Max.X > x; x++ {
			if masked(x, y) {
				continue
			}
			si, di := src.PixOffset(x, y), dst.PixOffset(x, y)
			assert.Equal(t, src.Pix[si:si+4], dst.Pix[di:di+4],
				"pixel outside the masked region changed at (%d,%d)", x, y)
		}
	}
}

func assertSourceImmutable(t *testing.T, src *image.NRGBA, before []uint8) {
	t.Helper()
	assert.Equal(t, before, src.Pix, "the source image pixel buffer was modified")
}

// TestAnonymize_PixelateHandComputed verifies pixelation against values computed
// by hand, including a block clipped by the region edge and integer floor division.
func TestAnonymize_PixelateHandComputed(t *testing.T) {
	// 6x4 image, full region, block size 4:
	// block (0..4, 0..4) holds 16 pixels, R sum 264 -> 264/16 = 16
	// block (4..6, 0..4) holds 8 pixels,  R sum 156 -> 156/8  = 19
	src := buildPatternImage(6, 4)
	before := append([]uint8(nil), src.Pix...)

	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 6, 4)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 4})
	require.NoError(t, err)
	require.Equal(t, src.Bounds(), dst.Bounds())

	for y := 0; y < 4; y++ {
		for x := 0; x < 6; x++ {
			i := dst.PixOffset(x, y)
			wantR := uint8(16)
			if x >= 4 {
				wantR = 19
			}
			assert.Equal(t, wantR, dst.Pix[i+0], "R at (%d,%d)", x, y)
			assert.Equal(t, uint8(100), dst.Pix[i+1], "G at (%d,%d)", x, y)
			assert.Equal(t, uint8(200), dst.Pix[i+2], "B at (%d,%d)", x, y)
		}
	}
	assertAlphaPreserved(t, src, dst)
	assertSourceImmutable(t, src, before)
}

// TestAnonymize_PixelateSingleBlock covers the straightforward whole-region average.
func TestAnonymize_PixelateSingleBlock(t *testing.T) {
	// 5x2 region: R values 0..4 and 10..14, sum 70 over 10 pixels -> 7.
	src := buildPatternImage(5, 4)
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 5, 2)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 5})
	require.NoError(t, err)

	for y := 0; y < 2; y++ {
		for x := 0; x < 5; x++ {
			i := dst.PixOffset(x, y)
			assert.Equal(t, uint8(7), dst.Pix[i+0], "R at (%d,%d)", x, y)
		}
	}
	// Rows 2 and 3 must be byte-identical to the source.
	assertOutsideUntouched(t, src, dst, []image.Rectangle{image.Rect(0, 0, 5, 2)})
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_BlurHandComputed checks the separable box blur against a
// hand-computed 3x3 result with border clamping.
func TestAnonymize_BlurHandComputed(t *testing.T) {
	src := buildCustomImage([][]uint8{
		{0, 0, 0},
		{0, 100, 0},
		{0, 0, 0},
	})
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 3, 3)},
		pigo.AnonymizeConfig{Method: pigo.Blur, BlurRadius: 1})
	require.NoError(t, err)

	want := [][]uint8{
		{25, 16, 25},
		{16, 11, 16},
		{25, 16, 25},
	}
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			assert.Equal(t, want[y][x], dst.Pix[dst.PixOffset(x, y)+0],
				"blurred R at (%d,%d)", x, y)
		}
	}
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_BlurSingleMaskedRow exercises horizontal clamping plus the
// vertical pass averaging over the single masked row.
func TestAnonymize_BlurSingleMaskedRow(t *testing.T) {
	// 5x3 image, only the middle row (R = 10..14) is masked.
	src := buildPatternImage(5, 3)
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 1, 5, 2)},
		pigo.AnonymizeConfig{Method: pigo.Blur, BlurRadius: 1})
	require.NoError(t, err)

	wantR := []uint8{10, 11, 12, 13, 13}
	for x := 0; x < 5; x++ {
		assert.Equal(t, wantR[x], dst.Pix[dst.PixOffset(x, 1)],
			"blurred R at (%d,1)", x)
	}
	assertOutsideUntouched(t, src, dst, []image.Rectangle{image.Rect(0, 1, 5, 2)})
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_OverlappingFacesMerged ensures overlapping faces are processed
// once: the union becomes a single block and repeated runs are byte-identical.
func TestAnonymize_OverlappingFacesMerged(t *testing.T) {
	src := buildPatternImage(8, 8)
	rects := []image.Rectangle{
		image.Rect(1, 1, 5, 5),
		image.Rect(3, 3, 7, 7), // overlaps the first one
	}
	cfg := pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 20}

	dst1, err := pigo.Anonymize(src, rects, cfg)
	require.NoError(t, err)
	dst2, err := pigo.Anonymize(src, rects, cfg)
	require.NoError(t, err)
	assert.Equal(t, dst1.Pix, dst2.Pix, "repeated runs must be deterministic")

	// The union (1,1)-(7,7) is one block: every masked pixel shares one average.
	first := dst1.Pix[dst1.PixOffset(1, 1) : dst1.PixOffset(1, 1)+3]
	for y := 1; y < 7; y++ {
		for x := 1; x < 7; x++ {
			i := dst1.PixOffset(x, y)
			assert.Equal(t, first, dst1.Pix[i:i+3], "union pixel at (%d,%d) diverges", x, y)
		}
	}
	assertOutsideUntouched(t, src, dst1, []image.Rectangle{image.Rect(1, 1, 7, 7)})
	assertAlphaPreserved(t, src, dst1)
}

// TestAnonymize_EdgeFaceClipping feeds negative and out-of-bounds coordinates;
// rectangles are clipped to the bounds, zero-area and fully out-of-bounds
// rectangles are skipped, and merely touching regions stay independent.
func TestAnonymize_EdgeFaceClipping(t *testing.T) {
	src := buildPatternImage(6, 6)
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{
			image.Rect(-2, -2, 3, 3),       // clipped to (0,0)-(3,3)
			image.Rect(4, 4, 100, 100),     // clipped to (4,4)-(6,6)
			image.Rect(100, 100, 200, 200), // fully outside, skipped
			image.Rect(2, 2, 2, 5),         // zero area, skipped
		},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 10})
	require.NoError(t, err)

	// First region (0,0)-(3,3): R values sum to 99 over 9 pixels -> 11.
	for y := 0; y < 3; y++ {
		for x := 0; x < 3; x++ {
			assert.Equal(t, uint8(11), dst.Pix[dst.PixOffset(x, y)],
				"clipped region pixel at (%d,%d)", x, y)
		}
	}
	// Second region: R values 44,45 / 54,55 sum to 198 over 4 pixels -> 49.
	for y := 4; y < 6; y++ {
		for x := 4; x < 6; x++ {
			assert.Equal(t, uint8(49), dst.Pix[dst.PixOffset(x, y)],
				"clipped region pixel at (%d,%d)", x, y)
		}
	}
	want := []image.Rectangle{
		image.Rect(0, 0, 3, 3),
		image.Rect(4, 4, 6, 6),
	}
	assertOutsideUntouched(t, src, dst, want)
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_WhitelistExcludesPixels verifies whitelisted pixels keep their
// original values and are excluded from the block average.
func TestAnonymize_WhitelistExcludesPixels(t *testing.T) {
	// 4x4 full image, one block of 16 pixels, R sum 264. The whitelisted pixel
	// (1,1) has R=11, so the remaining 15 pixels average (264-11)/15 = 16.
	src := buildPatternImage(4, 4)
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 4, 4)},
		pigo.AnonymizeConfig{
			Method:    pigo.Pixelate,
			BlockSize: 4,
			Whitelist: []image.Rectangle{image.Rect(1, 1, 2, 2)},
		})
	require.NoError(t, err)

	si := src.PixOffset(1, 1)
	di := dst.PixOffset(1, 1)
	assert.Equal(t, src.Pix[si:si+4], dst.Pix[di:di+4], "whitelisted pixel was modified")

	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if x == 1 && y == 1 {
				continue
			}
			assert.Equal(t, uint8(16), dst.Pix[dst.PixOffset(x, y)],
				"masked pixel at (%d,%d)", x, y)
		}
	}
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_MarginExpandsAndShrinks checks the configurable margin,
// including a negative margin that shrinks the region.
func TestAnonymize_MarginExpandsAndShrinks(t *testing.T) {
	src := buildPatternImage(6, 6)

	expanded, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(2, 2, 4, 4)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 10, Margin: 1})
	require.NoError(t, err)
	assertOutsideUntouched(t, src, expanded, []image.Rectangle{image.Rect(1, 1, 5, 5)})

	shrunk, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(2, 2, 4, 4)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 10, Margin: -1})
	require.NoError(t, err)
	// Shrinking by 1 on every side leaves the single pixel (3,3).
	assertOutsideUntouched(t, src, shrunk, []image.Rectangle{image.Rect(3, 3, 4, 4)})
	// A one-pixel block averages to its own value, hence everything stays equal.
	assert.Equal(t, src.Pix, shrunk.Pix)
}

// TestAnonymize_NoRegionsReturnsEqualCopy covers zero-area / out-of-bounds-only
// inputs: the result is a fresh, unchanged copy.
func TestAnonymize_NoRegionsReturnsEqualCopy(t *testing.T) {
	src := buildPatternImage(4, 4)
	dst, err := pigo.Anonymize(src,
		[]image.Rectangle{
			image.Rect(10, 10, 20, 20),
			image.Rect(0, 0, 0, 0),
		},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 4})
	require.NoError(t, err)
	assert.Equal(t, src.Pix, dst.Pix)
	assert.NotSame(t, src, dst)
}

// TestAnonymize_NonZeroBounds makes sure absolute rectangle coordinates are
// handled correctly when the image bounds do not start at the origin.
func TestAnonymize_NonZeroBounds(t *testing.T) {
	rect := image.Rect(10, 10, 14, 14)
	img := image.NewNRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8(x)
			img.Pix[i+1] = uint8(y)
			img.Pix[i+2] = 42
			img.Pix[i+3] = 255
		}
	}
	dst, err := pigo.Anonymize(img,
		[]image.Rectangle{image.Rect(9, 9, 20, 20)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 10})
	require.NoError(t, err)
	require.Equal(t, rect, dst.Bounds())

	// R average of x = 10+11+12+13 = 46/4 = 11, G average of y = 46/4 = 11.
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			i := dst.PixOffset(x, y)
			assert.Equal(t, uint8(11), dst.Pix[i+0])
			assert.Equal(t, uint8(11), dst.Pix[i+1])
			assert.Equal(t, uint8(42), dst.Pix[i+2])
			assert.Equal(t, uint8(255), dst.Pix[i+3])
		}
	}
}

// TestAnonymizeDetections_FiltersByConfidenceAndSize exercises the detection
// based entry point, including the Detection -> square rectangle conversion.
func TestAnonymizeDetections_FiltersByConfidenceAndSize(t *testing.T) {
	src := buildPatternImage(8, 8)
	dets := []pigo.Detection{
		{Row: 2, Col: 2, Scale: 4, Q: 0.9},  // square (0,0)-(4,4), accepted
		{Row: 2, Col: 2, Scale: 4, Q: 0.2},  // below confidence, skipped
		{Row: 6, Col: 6, Scale: 2, Q: 0.95}, // below min face size, skipped
	}
	dst, err := pigo.AnonymizeDetections(src, dets, pigo.AnonymizeConfig{
		Method:        pigo.Pixelate,
		BlockSize:     4,
		MinConfidence: 0.5,
		MinFaceSize:   3,
	})
	require.NoError(t, err)

	// Only the (0,0)-(4,4) square is masked; its block average R =
	// sum(10y+x, x,y in 0..3) = 264 / 16 = 16.
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			assert.Equal(t, uint8(16), dst.Pix[dst.PixOffset(x, y)],
				"detected face pixel at (%d,%d)", x, y)
		}
	}
	assertOutsideUntouched(t, src, dst, []image.Rectangle{image.Rect(0, 0, 4, 4)})
	assertAlphaPreserved(t, src, dst)
}

// TestAnonymize_ErrorCases enumerates the explicit handling required for
// degenerate inputs and illegal configuration values.
func TestAnonymize_ErrorCases(t *testing.T) {
	opaque := buildPatternImage(4, 4)
	transparent := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	zeroSize := image.NewNRGBA(image.Rect(0, 0, 0, 0))
	region := []image.Rectangle{image.Rect(0, 0, 4, 4)}

	tests := []struct {
		name string
		img  *image.NRGBA
		cfg  pigo.AnonymizeConfig
		err  error
	}{
		{"nil image", nil, pigo.AnonymizeConfig{Method: pigo.Pixelate}, pigo.ErrNilImage},
		{"zero dimensions", zeroSize, pigo.AnonymizeConfig{Method: pigo.Pixelate}, pigo.ErrEmptyImage},
		{"fully transparent", transparent, pigo.AnonymizeConfig{Method: pigo.Pixelate}, pigo.ErrFullyTransparent},
		{"image exceeds pixel cap", opaque, pigo.AnonymizeConfig{
			Method: pigo.Pixelate, MaxPixels: 10,
		}, pigo.ErrImageTooLarge},
		{"negative block size", opaque, pigo.AnonymizeConfig{
			Method: pigo.Pixelate, BlockSize: -1,
		}, pigo.ErrInvalidBlockSize},
		{"negative blur radius", opaque, pigo.AnonymizeConfig{
			Method: pigo.Blur, BlurRadius: -1,
		}, pigo.ErrInvalidBlurRadius},
		{"unknown method", opaque, pigo.AnonymizeConfig{
			Method: pigo.AnonymizeMethod(99),
		}, pigo.ErrInvalidMethod},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, anErr := pigo.Anonymize(tc.img, region, tc.cfg)
			assert.ErrorIs(t, anErr, tc.err)
		})
	}
}

// TestAnonymize_DefaultsApplied confirms zero-valued tuning fields fall back to
// the documented defaults instead of failing or dividing by zero.
func TestAnonymize_DefaultsApplied(t *testing.T) {
	src := buildPatternImage(8, 8)

	pix, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 8, 8)},
		pigo.AnonymizeConfig{Method: pigo.Pixelate})
	require.NoError(t, err)
	require.NotEqual(t, src.Pix, pix.Pix)

	blurred, err := pigo.Anonymize(src,
		[]image.Rectangle{image.Rect(0, 0, 8, 8)},
		pigo.AnonymizeConfig{Method: pigo.Blur})
	require.NoError(t, err)
	require.NotEqual(t, src.Pix, blurred.Pix)
	assertAlphaPreserved(t, src, blurred)
}

// TestAnonymize_CascadeIntegration runs the original face detection pipeline on
// the repository sample image and anonymizes the clustered detections, proving
// the new feature composes with the existing cascade output.
func TestAnonymize_CascadeIntegration(t *testing.T) {
	classifier := p.Unpack()
	require.NotNil(t, classifier)

	dets := classifier.RunCascade(*cParams, 0.0)
	dets = classifier.ClusterDetections(dets, 0.1)
	require.NotEmpty(t, dets)

	dst, err := pigo.AnonymizeDetections(srcImg, dets, pigo.AnonymizeConfig{
		Method:        pigo.Pixelate,
		BlockSize:     12,
		MinConfidence: 0.0,
		Margin:        4,
	})
	require.NoError(t, err)
	require.Equal(t, srcImg.Bounds(), dst.Bounds())
	assert.NotEqual(t, srcImg.Pix, dst.Pix, "expected at least one anonymized face")

	// Alpha must be preserved even on real-world images.
	assertAlphaPreserved(t, srcImg, dst)
}

func BenchmarkAnonymizePixelate(b *testing.B) {
	src := buildPatternImage(500, 500)
	rects := []image.Rectangle{
		image.Rect(50, 50, 200, 200),
		image.Rect(180, 180, 460, 460),
	}
	cfg := pigo.AnonymizeConfig{Method: pigo.Pixelate, BlockSize: 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = pigo.Anonymize(src, rects, cfg)
	}
}
