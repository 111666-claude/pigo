package pigo_test

import (
	"errors"
	"image"
	"image/color"
	"math"
	"testing"

	pigo "github.com/esimov/pigo/pkg"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newSolidNRGBA returns a w x h image with the given bounds origin filled with c.
func newSolidNRGBA(rect image.Rectangle, c color.NRGBA) *image.NRGBA {
	img := image.NewNRGBA(rect)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

// fillDisk paints an axis-aligned filled disk into img.
func fillDisk(img *image.NRGBA, cx, cy, radius int, c color.NRGBA) {
	for y := cy - radius; y <= cy+radius; y++ {
		for x := cx - radius; x <= cx+radius; x++ {
			dx, dy := x-cx, y-cy
			if dx*dx+dy*dy <= radius*radius && image.Pt(x, y).In(img.Bounds()) {
				img.SetNRGBA(x, y, c)
			}
		}
	}
}

func assertPointClose(t *testing.T, want, got pigo.Point, tol float64) {
	t.Helper()
	assert.InDeltaf(t, want.X, got.X, tol, "X coordinate: want %v, got %v", want, got)
	assert.InDeltaf(t, want.Y, got.Y, tol, "Y coordinate: want %v, got %v", want, got)
}

func assertColorClose(t *testing.T, want, got color.NRGBA, tol uint8) {
	t.Helper()
	assert.InDeltaf(t, int(want.R), int(got.R), float64(tol), "red: want %v, got %v", want, got)
	assert.InDeltaf(t, int(want.G), int(got.G), float64(tol), "green: want %v, got %v", want, got)
	assert.InDeltaf(t, int(want.B), int(got.B), float64(tol), "blue: want %v, got %v", want, got)
	assert.InDeltaf(t, int(want.A), int(got.A), float64(tol), "alpha: want %v, got %v", want, got)
}

// TestAlignFaceTiltedEyes verifies that a tilted pair of eyes is brought onto
// a horizontal line at the configured eye distance and anchor, and that a
// non-eye landmark follows the same rigid transform.
func TestAlignFaceTiltedEyes(t *testing.T) {
	src := newSolidNRGBA(image.Rect(0, 0, 160, 140), color.NRGBA{30, 30, 30, 255})

	left := pigo.Point{X: 50, Y: 60}
	right := pigo.Point{X: 110, Y: 85}
	mid := pigo.Point{X: (left.X + right.X) / 2, Y: (left.Y + right.Y) / 2}

	// Distinctive markers: eyes plus an off-axis landmark above the face.
	leftColor := color.NRGBA{255, 0, 0, 255}
	rightColor := color.NRGBA{0, 255, 0, 255}
	foreheadColor := color.NRGBA{0, 128, 255, 255}
	fillDisk(src, int(left.X), int(left.Y), 5, leftColor)
	fillDisk(src, int(right.X), int(right.Y), 5, rightColor)
	fillDisk(src, 80, 40, 5, foreheadColor)

	const (
		targetDistance = 90.0
		outW, outH     = 120, 120
	)
	aligned, m, err := pigo.AlignFace(src, pigo.AlignOptions{
		Face:              image.Rect(10, 10, 150, 130),
		LeftEye:           left,
		RightEye:          right,
		TargetEyeDistance: targetDistance,
		Width:             outW,
		Height:            outH,
		Interpolation:     pigo.Bilinear,
	})
	require.NoError(t, err)
	require.NotNil(t, aligned)
	assert.Equal(t, image.Rect(0, 0, outW, outH), aligned.Bounds())

	anchor := pigo.Point{X: float64(outW-1) / 2, Y: float64(outH-1) / 2}
	half := targetDistance / 2

	// Requirement: the eyes land at fixed coordinates regardless of head tilt.
	assertPointClose(t, pigo.Point{X: anchor.X - half, Y: anchor.Y}, m.Apply(left), 1e-9)
	assertPointClose(t, pigo.Point{X: anchor.X + half, Y: anchor.Y}, m.Apply(right), 1e-9)
	assertPointClose(t, anchor, m.Apply(mid), 1e-9)

	// The pixel at the mapped eye centers must sample the eye markers.
	wantLeft := m.Apply(left)
	wantRight := m.Apply(right)
	gotLeft := aligned.NRGBAAt(int(math.Round(wantLeft.X)), int(math.Round(wantLeft.Y)))
	gotRight := aligned.NRGBAAt(int(math.Round(wantRight.X)), int(math.Round(wantRight.Y)))
	assertColorClose(t, leftColor, gotLeft, 2)
	assertColorClose(t, rightColor, gotRight, 2)

	// Same for the off-axis landmark: this validates rotation, not only the
	// two eye constraints.
	wantForehead := m.Apply(pigo.Point{X: 80, Y: 40})
	require.True(t, image.Pt(int(math.Round(wantForehead.X)), int(math.Round(wantForehead.Y))).In(aligned.Bounds()))
	gotForehead := aligned.NRGBAAt(int(math.Round(wantForehead.X)), int(math.Round(wantForehead.Y)))
	assertColorClose(t, foreheadColor, gotForehead, 8)

	// The transform must be invertible and round-trip every probe point.
	inv, err := m.Invert()
	require.NoError(t, err)
	for _, p := range []pigo.Point{
		left, right, mid,
		{X: 0, Y: 0}, {X: 159, Y: 139}, {X: 80, Y: 40},
	} {
		assertPointClose(t, p, inv.Apply(m.Apply(p)), 1e-9)
	}

	// Source image must not be modified.
	clean := newSolidNRGBA(image.Rect(0, 0, 160, 140), color.NRGBA{30, 30, 30, 255})
	fillDisk(clean, int(left.X), int(left.Y), 5, leftColor)
	fillDisk(clean, int(right.X), int(right.Y), 5, rightColor)
	fillDisk(clean, 80, 40, 5, foreheadColor)
	assert.Equal(t, clean.Pix, src.Pix)
}

// TestAlignFaceEyeAnchorAndNonZeroBounds checks an explicit eye anchor and a
// source image whose Bounds().Min is not the origin.
func TestAlignFaceEyeAnchorAndNonZeroBounds(t *testing.T) {
	const (
		minX, minY = 10, 20
		w, h       = 100, 80
	)
	src := image.NewNRGBA(image.Rect(minX, minY, minX+w, minY+h))
	for y := minY; y < minY+h; y++ {
		for x := minX; x < minX+w; x++ {
			src.SetNRGBA(x, y, color.NRGBA{uint8(x * 2), uint8(y * 2), 77, 255})
		}
	}

	left := pigo.Point{X: minX + 20, Y: minY + 30}
	right := pigo.Point{X: minX + 60, Y: minY + 30}
	anchor := pigo.Point{X: 40, Y: 50}
	aligned, m, err := pigo.AlignFace(src, pigo.AlignOptions{
		Face:              image.Rect(minX, minY, minX+w, minY+h),
		LeftEye:           left,
		RightEye:          right,
		TargetEyeDistance: 80,
		Width:             96,
		Height:            96,
		EyeAnchor:         &anchor,
		Interpolation:     pigo.NearestNeighbor,
	})
	require.NoError(t, err)

	assertPointClose(t, pigo.Point{X: anchor.X - 40, Y: anchor.Y}, m.Apply(left), 1e-9)
	assertPointClose(t, pigo.Point{X: anchor.X + 40, Y: anchor.Y}, m.Apply(right), 1e-9)

	// Eyes are horizontal, so this is a pure scale+translation: the source
	// pixel at the anchor midpoint maps to the anchor itself.
	assertPointClose(t, anchor, m.Apply(pigo.Point{X: left.X + 20, Y: left.Y}), 1e-9)

	got := aligned.NRGBAAt(int(anchor.X), int(anchor.Y))
	want := src.NRGBAAt(int(left.X+20), int(left.Y))
	assertColorClose(t, want, got, 1)

	// A source point far from the eyes maps deterministically.
	probe := pigo.Point{X: minX, Y: minY}
	mapped := m.Apply(probe)
	assertPointClose(t, probe, mustInvert(t, m).Apply(mapped), 1e-9)
}

func mustInvert(t *testing.T, m pigo.AffineMatrix) pigo.AffineMatrix {
	t.Helper()
	inv, err := m.Invert()
	require.NoError(t, err)
	return inv
}

// TestAlignFaceIdentity checks that a unit-scale, tilt-free alignment with
// integer sampling reproduces the source exactly, for several image types and
// non-zero bounds.
func TestAlignFaceIdentity(t *testing.T) {
	t.Run("NRGBA-nonzero-bounds", func(t *testing.T) {
		rect := image.Rect(10, 20, 14, 24)
		src := image.NewNRGBA(rect)
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			for x := rect.Min.X; x < rect.Max.X; x++ {
				src.SetNRGBA(x, y, color.NRGBA{uint8(x), uint8(y), uint8(x + y), 255})
			}
		}

		aligned, m, err := pigo.AlignFace(src, pigo.AlignOptions{
			Face:              rect,
			LeftEye:           pigo.Point{X: float64(rect.Min.X) + 1, Y: float64(rect.Min.Y) + 1.5},
			RightEye:          pigo.Point{X: float64(rect.Min.X) + 2, Y: float64(rect.Min.Y) + 1.5},
			TargetEyeDistance: 1,
			Width:             rect.Dx(),
			Height:            rect.Dy(),
			Interpolation:     pigo.Bilinear,
		})
		require.NoError(t, err)

		for y := 0; y < rect.Dy(); y++ {
			for x := 0; x < rect.Dx(); x++ {
				want := src.NRGBAAt(x+rect.Min.X, y+rect.Min.Y)
				got := aligned.NRGBAAt(x, y)
				assert.Equal(t, want, got, "pixel (%d,%d)", x, y)
				// The inverse must map output pixels back onto source pixels.
				back := mustInvert(t, m).Apply(pigo.Point{X: float64(x), Y: float64(y)})
				assertPointClose(t, pigo.Point{X: float64(x + rect.Min.X), Y: float64(y + rect.Min.Y)}, back, 1e-9)
			}
		}
	})

	t.Run("Gray-nonzero-bounds", func(t *testing.T) {
		rect := image.Rect(-6, 3, -2, 7)
		src := image.NewGray(rect)
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			for x := rect.Min.X; x < rect.Max.X; x++ {
				src.SetGray(x, y, color.Gray{Y: uint8((x-rect.Min.X)*4 + (y-rect.Min.Y)*17)})
			}
		}

		aligned, _, err := pigo.AlignFace(src, pigo.AlignOptions{
			Face:              rect,
			LeftEye:           pigo.Point{X: float64(rect.Min.X) + 1, Y: float64(rect.Min.Y) + 1.5},
			RightEye:          pigo.Point{X: float64(rect.Min.X) + 2, Y: float64(rect.Min.Y) + 1.5},
			TargetEyeDistance: 1,
			Width:             rect.Dx(),
			Height:            rect.Dy(),
			Interpolation:     pigo.NearestNeighbor,
		})
		require.NoError(t, err)

		for y := 0; y < rect.Dy(); y++ {
			for x := 0; x < rect.Dx(); x++ {
				v := src.GrayAt(x+rect.Min.X, y+rect.Min.Y).Y
				assertColorClose(t, color.NRGBA{R: v, G: v, B: v, A: 255}, aligned.NRGBAAt(x, y), 0)
			}
		}
	})

	t.Run("RGBA", func(t *testing.T) {
		rect := image.Rect(0, 0, 4, 4)
		src := image.NewRGBA(rect)
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				src.SetRGBA(x, y, color.RGBA{uint8(20 * x), uint8(30 * y), 200, 255})
			}
		}

		aligned, _, err := pigo.AlignFace(src, pigo.AlignOptions{
			Face:              rect,
			LeftEye:           pigo.Point{X: 1, Y: 1.5},
			RightEye:          pigo.Point{X: 2, Y: 1.5},
			TargetEyeDistance: 1,
			Width:             4,
			Height:            4,
			Interpolation:     pigo.Bilinear,
		})
		require.NoError(t, err)

		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				want := color.NRGBAModel.Convert(src.At(x, y)).(color.NRGBA)
				assert.Equal(t, want, aligned.NRGBAAt(x, y))
			}
		}
	})
}

// TestAlignFaceInterpolation pins down the exact resampled values for a known
// scale/translation at both fractional and edge-overlapping output pixels.
func TestAlignFaceInterpolation(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	src.SetNRGBA(0, 0, color.NRGBA{10, 20, 30, 255})
	src.SetNRGBA(1, 0, color.NRGBA{110, 120, 130, 255})
	src.SetNRGBA(0, 1, color.NRGBA{210, 220, 230, 255})
	src.SetNRGBA(1, 1, color.NRGBA{250, 250, 250, 255})

	opts := pigo.AlignOptions{
		Face:              image.Rect(0, 0, 4, 4),
		LeftEye:           pigo.Point{X: 1, Y: 1},
		RightEye:          pigo.Point{X: 2, Y: 1},
		TargetEyeDistance: 2,
		Width:             8,
		Height:            8,
	}
	// With these options: x' = 2x + 0.5, y' = 2y + 1.5, so output (1,2)
	// samples the source at (0.25, 0.25) and output (0,2) at (-0.25, 0.25).

	t.Run("bilinear interior", func(t *testing.T) {
		o := opts
		aligned, _, err := pigo.AlignFace(src, o)
		require.NoError(t, err)
		// Weights: p00*0.5625 + p10*0.1875 + p01*0.1875 + p11*0.0625.
		want := color.NRGBA{81, 91, 100, 255}
		assertColorClose(t, want, aligned.NRGBAAt(1, 2), 1)
	})

	t.Run("nearest interior", func(t *testing.T) {
		o := opts
		o.Interpolation = pigo.NearestNeighbor
		aligned, _, err := pigo.AlignFace(src, o)
		require.NoError(t, err)
		assert.Equal(t, color.NRGBA{10, 20, 30, 255}, aligned.NRGBAAt(1, 2))
	})

	t.Run("bilinear blends background at the edge", func(t *testing.T) {
		o := opts
		o.Background = color.White
		aligned, _, err := pigo.AlignFace(src, o)
		require.NoError(t, err)
		// Output (0,2) samples (-0.25, 0.25): x0 = -1, so the background
		// weights are 0.1875 (top-left) and 0.0625 (bottom-left), source
		// (0,0) gets 0.5625 and source (0,1) gets 0.1875. The background is
		// opaque white: 255*0.25 + 10*0.5625 + 210*0.1875 = 108.75.
		want := color.NRGBA{109, 116, 124, 255}
		assertColorClose(t, want, aligned.NRGBAAt(0, 2), 1)
	})
}

// TestAlignFaceFillsOutsideSource verifies transparent and solid background
// fill for a crop that partially and fully leaves the source image.
func TestAlignFaceFillsOutsideSource(t *testing.T) {
	// Face and eyes sit at the left edge of a small image, so the right part
	// of the aligned crop falls outside the source.
	src := newSolidNRGBA(image.Rect(0, 0, 20, 20), color.NRGBA{255, 255, 255, 255})
	opts := pigo.AlignOptions{
		Face:              image.Rect(0, 0, 20, 20),
		LeftEye:           pigo.Point{X: 2, Y: 10},
		RightEye:          pigo.Point{X: 6, Y: 10},
		TargetEyeDistance: 10, // scale 2.5: the output is wider than the source
		Width:             64,
		Height:            64,
	}

	transparent, _, err := pigo.AlignFace(src, opts)
	require.NoError(t, err)
	// Corners map well outside the source and must be fully transparent.
	for _, p := range []image.Point{{0, 0}, {63, 0}, {0, 63}, {63, 63}} {
		assert.Equal(t, uint8(0), transparent.NRGBAAt(p.X, p.Y).A, "corner %v", p)
	}
	// The eye midpoint maps to the anchor, which is inside the source.
	anchor := image.Pt(31, 31)
	assert.Equal(t, color.NRGBA{255, 255, 255, 255}, transparent.NRGBAAt(anchor.X, anchor.Y))

	bg := color.NRGBA{12, 34, 56, 255}
	o := opts
	o.Background = bg
	filled, _, err := pigo.AlignFace(src, o)
	require.NoError(t, err)
	for _, p := range []image.Point{{0, 0}, {63, 0}, {0, 63}, {63, 63}} {
		assert.Equal(t, bg, filled.NRGBAAt(p.X, p.Y), "corner %v", p)
	}
	assert.Equal(t, color.NRGBA{255, 255, 255, 255}, filled.NRGBAAt(anchor.X, anchor.Y))

	// With the anchor outside the output, every pixel leaves the source.
	far := pigo.Point{X: -1000, Y: -1000}
	o = opts
	o.EyeAnchor = &far
	o.Background = color.Black
	fullyOutside, _, err := pigo.AlignFace(src, o)
	require.NoError(t, err)
	for y := 0; y < o.Height; y++ {
		for x := 0; x < o.Width; x++ {
			assert.Equal(t, color.NRGBA{0, 0, 0, 255}, fullyOutside.NRGBAAt(x, y), "pixel (%d,%d)", x, y)
		}
	}
}

// TestAlignFaceEdgeFaceTiltedEyes combines a face rectangle extending past the
// image border with a tilted eye pair.
func TestAlignFaceEdgeFaceTiltedEyes(t *testing.T) {
	src := newSolidNRGBA(image.Rect(0, 0, 24, 24), color.NRGBA{200, 180, 160, 255})
	left := pigo.Point{X: 4, Y: 10}
	right := pigo.Point{X: 12, Y: 14}
	leftColor := color.NRGBA{255, 0, 0, 255}
	rightColor := color.NRGBA{0, 255, 0, 255}
	fillDisk(src, int(left.X), int(left.Y), 2, leftColor)
	fillDisk(src, int(right.X), int(right.Y), 2, rightColor)

	aligned, m, err := pigo.AlignFace(src, pigo.AlignOptions{
		Face:              image.Rect(-10, -10, 30, 30), // extends beyond the image
		LeftEye:           left,
		RightEye:          right,
		TargetEyeDistance: 40,
		Width:             120,
		Height:            120,
		Interpolation:     pigo.NearestNeighbor,
		Background:        color.NRGBA{0, 0, 255, 255},
	})
	require.NoError(t, err)

	anchor := pigo.Point{X: 59.5, Y: 59.5}
	assertPointClose(t, pigo.Point{X: anchor.X - 20, Y: anchor.Y}, m.Apply(left), 1e-9)
	assertPointClose(t, pigo.Point{X: anchor.X + 20, Y: anchor.Y}, m.Apply(right), 1e-9)

	// The rotation is significant: the left eye's output neighborhood must be
	// red and the right eye's green, despite the tilt.
	assertColorClose(t, leftColor, aligned.NRGBAAt(40, 59), 20)
	assertColorClose(t, rightColor, aligned.NRGBAAt(81, 60), 20)

	// Source rectangle maps to the middle of the output; the far corners are
	// outside and must be filled with the requested background.
	assert.Equal(t, color.NRGBA{0, 0, 255, 255}, aligned.NRGBAAt(0, 0))
	assert.Equal(t, color.NRGBA{0, 0, 255, 255}, aligned.NRGBAAt(119, 119))

	// Both eyes map inside the output; the source is untouched.
	for _, p := range []pigo.Point{left, right} {
		out := m.Apply(p)
		assert.True(t, image.Pt(int(out.X), int(out.Y)).In(aligned.Bounds()))
	}
}

// TestAlignFaceExtremeScale checks that a huge but permitted scale has a
// defined, non-panicking result, and that a scale beyond the limit is rejected.
func TestAlignFaceExtremeScale(t *testing.T) {
	src := newSolidNRGBA(image.Rect(0, 0, 4, 4), color.NRGBA{70, 80, 90, 255})
	opts := pigo.AlignOptions{
		Face:              image.Rect(0, 0, 4, 4),
		LeftEye:           pigo.Point{X: 1.5, Y: 1},
		RightEye:          pigo.Point{X: 2.5, Y: 1},
		TargetEyeDistance: pigo.MaxAlignScale, // distance is 1, so scale == limit
		Width:             8,
		Height:            8,
		Interpolation:     pigo.Bilinear,
	}
	aligned, _, err := pigo.AlignFace(src, opts)
	require.NoError(t, err)
	require.Equal(t, 8, aligned.Bounds().Dx())
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			// The sampled area is microscopic but still inside the solid source.
			assertColorClose(t, color.NRGBA{70, 80, 90, 255}, aligned.NRGBAAt(x, y), 1)
		}
	}

	o := opts
	o.TargetEyeDistance = pigo.MaxAlignScale + 1
	_, _, err = pigo.AlignFace(src, o)
	assert.ErrorIs(t, err, pigo.ErrAlignScaleOverflow)
}

func TestAlignFaceErrors(t *testing.T) {
	src := newSolidNRGBA(image.Rect(0, 0, 32, 32), color.NRGBA{1, 2, 3, 255})

	base := func() pigo.AlignOptions {
		return pigo.AlignOptions{
			Face:              image.Rect(0, 0, 32, 32),
			LeftEye:           pigo.Point{X: 10, Y: 10},
			RightEye:          pigo.Point{X: 20, Y: 10},
			TargetEyeDistance: 10,
			Width:             32,
			Height:            32,
		}
	}

	t.Run("nil image", func(t *testing.T) {
		_, _, err := pigo.AlignFace(nil, base())
		assert.ErrorIs(t, err, pigo.ErrAlignNilImage)
	})

	t.Run("empty image", func(t *testing.T) {
		empty := image.NewNRGBA(image.Rect(5, 5, 5, 5))
		_, _, err := pigo.AlignFace(empty, base())
		assert.ErrorIs(t, err, pigo.ErrAlignEmptyImage)
	})

	tests := []struct {
		name   string
		mutate func(*pigo.AlignOptions)
		want   error
	}{
		{"zero face rectangle", func(o *pigo.AlignOptions) { o.Face = image.Rect(10, 10, 10, 20) }, pigo.ErrAlignInvalidFaceRect},
		{"zero output width", func(o *pigo.AlignOptions) { o.Width = 0 }, pigo.ErrAlignInvalidOutputSize},
		{"negative output height", func(o *pigo.AlignOptions) { o.Height = -3 }, pigo.ErrAlignInvalidOutputSize},
		{"output too large", func(o *pigo.AlignOptions) { o.Width, o.Height = 10000, 10000 }, pigo.ErrAlignOutputTooLarge},
		{"absurd output width", func(o *pigo.AlignOptions) { o.Width = math.MaxInt }, pigo.ErrAlignOutputTooLarge},
		{"unknown interpolation", func(o *pigo.AlignOptions) { o.Interpolation = pigo.Interpolation(7) }, pigo.ErrAlignInvalidInterp},
		{"coincident eyes", func(o *pigo.AlignOptions) { o.RightEye = o.LeftEye }, pigo.ErrAlignCoincidentEyes},
		{"reversed eyes", func(o *pigo.AlignOptions) { o.LeftEye, o.RightEye = pigo.Point{X: 20, Y: 10}, pigo.Point{X: 10, Y: 10} }, pigo.ErrAlignReversedEyes},
		{"left eye outside face", func(o *pigo.AlignOptions) { o.Face = image.Rect(2, 2, 30, 30); o.LeftEye = pigo.Point{X: 1, Y: 10} }, pigo.ErrAlignEyesOutsideFace},
		{"right eye on face boundary", func(o *pigo.AlignOptions) { o.Face = image.Rect(0, 0, 20, 30) }, pigo.ErrAlignEyesOutsideFace},
		{"left eye NaN", func(o *pigo.AlignOptions) { o.LeftEye.X = math.NaN() }, pigo.ErrAlignNonFinitePoint},
		{"right eye inf", func(o *pigo.AlignOptions) { o.RightEye.Y = math.Inf(1) }, pigo.ErrAlignNonFinitePoint},
		{"anchor NaN", func(o *pigo.AlignOptions) { o.EyeAnchor = &pigo.Point{X: 1, Y: math.NaN()} }, pigo.ErrAlignNonFinitePoint},
		{"zero target distance", func(o *pigo.AlignOptions) { o.TargetEyeDistance = 0 }, pigo.ErrAlignInvalidEyeDistance},
		{"negative target distance", func(o *pigo.AlignOptions) { o.TargetEyeDistance = -1 }, pigo.ErrAlignInvalidEyeDistance},
		{"NaN target distance", func(o *pigo.AlignOptions) { o.TargetEyeDistance = math.NaN() }, pigo.ErrAlignInvalidEyeDistance},
		{"infinite target distance", func(o *pigo.AlignOptions) { o.TargetEyeDistance = math.Inf(-1) }, pigo.ErrAlignInvalidEyeDistance},
		{"scale overflow", func(o *pigo.AlignOptions) {
			o.LeftEye = pigo.Point{X: 10, Y: 10}
			o.RightEye = pigo.Point{X: 11, Y: 10}
			o.TargetEyeDistance = 1e12
		}, pigo.ErrAlignScaleOverflow},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			o := base()
			tc.mutate(&o)
			img, m, err := pigo.AlignFace(src, o)
			assert.Nil(t, img)
			assert.Equal(t, pigo.AffineMatrix{}, m)
			require.Error(t, err)
			assert.Truef(t, errors.Is(err, tc.want), "want %v, got %v", tc.want, err)
		})
	}

	// An eye exactly on the inclusive minimum is accepted.
	t.Run("eye on minimum boundary is valid", func(t *testing.T) {
		o := base()
		o.Face = image.Rect(2, 2, 30, 30)
		o.LeftEye = pigo.Point{X: 2, Y: 10}
		o.RightEye = pigo.Point{X: 20, Y: 10}
		_, _, err := pigo.AlignFace(src, o)
		assert.NoError(t, err)
	})
}

func TestAffineMatrixInvert(t *testing.T) {
	t.Run("singular", func(t *testing.T) {
		_, err := (pigo.AffineMatrix{A: 1, B: 2, D: 2, E: 4}).Invert()
		assert.ErrorIs(t, err, pigo.ErrAlignSingularTransform)
	})

	t.Run("zero matrix", func(t *testing.T) {
		_, err := (pigo.AffineMatrix{}).Invert()
		assert.ErrorIs(t, err, pigo.ErrAlignSingularTransform)
	})

	t.Run("overflowing", func(t *testing.T) {
		_, err := (pigo.AffineMatrix{A: 1e308, B: 1e308, D: 1e308, E: 1e308}).Invert()
		assert.ErrorIs(t, err, pigo.ErrAlignSingularTransform)
	})

	t.Run("round trip", func(t *testing.T) {
		m := pigo.AffineMatrix{
			A: 1.25, B: -0.5, C: 17,
			D: 0.75, E: 2.5, F: -8,
		}
		inv := mustInvert(t, m)
		for _, p := range []pigo.Point{
			{X: 0, Y: 0}, {X: 1, Y: 1}, {X: -3.5, Y: 9.25}, {X: 1e4, Y: -1e4},
		} {
			assertPointClose(t, p, inv.Apply(m.Apply(p)), 1e-9)
		}
	})
}

// TestAlignFaceDetectionResultsUnchanged runs the real detector over the
// sample image, aligns the detected face, and proves that neither the source
// pixels nor a repeated detection run changed.
func TestAlignFaceDetectionResultsUnchanged(t *testing.T) {
	classifier := pigo.NewPigo().Unpack()

	detections := classifier.RunCascade(*cParams, 0.0)
	detections = classifier.ClusterDetections(detections, 0.1)
	require.NotEmpty(t, detections)

	var det pigo.Detection
	for _, d := range detections {
		if d.Scale > 50 {
			det = d
			break
		}
	}
	require.NotZero(t, det.Scale, "no sufficiently large face was detected")

	detectionsBefore := append([]pigo.Detection(nil), detections...)
	pixelsBefore := append([]uint8(nil), srcImg.Pix...)

	half := det.Scale / 2
	face := image.Rect(det.Col-half, det.Row-half, det.Col+half, det.Row+half)
	left := pigo.Point{
		X: float64(det.Col) - 0.175*float64(det.Scale),
		Y: float64(det.Row) - 0.075*float64(det.Scale),
	}
	right := pigo.Point{
		X: float64(det.Col) + 0.185*float64(det.Scale),
		Y: float64(det.Row) - 0.075*float64(det.Scale),
	}

	const outW, outH = 112, 112
	aligned, m, err := pigo.AlignFace(srcImg, pigo.AlignOptions{
		Face:              face,
		LeftEye:           left,
		RightEye:          right,
		TargetEyeDistance: 70,
		Width:             outW,
		Height:            outH,
	})
	require.NoError(t, err)
	require.NotNil(t, aligned)

	anchor := pigo.Point{X: float64(outW-1) / 2, Y: float64(outH-1) / 2}
	assertPointClose(t, pigo.Point{X: anchor.X - 35, Y: anchor.Y}, m.Apply(left), 1e-9)
	assertPointClose(t, pigo.Point{X: anchor.X + 35, Y: anchor.Y}, m.Apply(right), 1e-9)

	// The aligned crop must contain real face pixels, not just background.
	opaque := 0
	for i := 3; i < len(aligned.Pix); i += 4 {
		if aligned.Pix[i] != 0 {
			opaque++
		}
	}
	assert.NotZero(t, opaque)

	detectionsAfter := classifier.RunCascade(*cParams, 0.0)
	detectionsAfter = classifier.ClusterDetections(detectionsAfter, 0.1)
	assert.Equal(t, detectionsBefore, detectionsAfter, "alignment must not alter detection results")
	assert.Equal(t, pixelsBefore, srcImg.Pix, "alignment must not alter the source image")
}

func BenchmarkAlignFace(b *testing.B) {
	src := newSolidNRGBA(image.Rect(0, 0, 320, 400), color.NRGBA{120, 130, 140, 255})
	opts := pigo.AlignOptions{
		Face:              image.Rect(40, 40, 280, 360),
		LeftEye:           pigo.Point{X: 120, Y: 160},
		RightEye:          pigo.Point{X: 200, Y: 175},
		TargetEyeDistance: 70,
		Width:             112,
		Height:            112,
	}

	for _, mode := range []struct {
		name   string
		interp pigo.Interpolation
	}{
		{"Bilinear", pigo.Bilinear},
		{"NearestNeighbor", pigo.NearestNeighbor},
	} {
		b.Run(mode.name, func(b *testing.B) {
			o := opts
			o.Interpolation = mode.interp
			b.ResetTimer()
			for b.Loop() {
				_, _, err := pigo.AlignFace(src, o)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
