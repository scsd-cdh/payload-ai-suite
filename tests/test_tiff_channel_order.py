"""Band order of 4-band images, from the file on disk to the model input.

4-band files store NIR, Red, Green, Blue (the Sentinel-2 download in fetch.py
asks for B08, B04, B03, B02). Every image must reach preprocessing as
preprocess.BAND_ORDER (Blue, Green, Red, NIR).

All 4-band files in data/labeled are PNG data, including the ones named
.tiff, so the fixtures write PNG data under both names. OpenCV picks the
decoder from the file contents, not the extension.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

import preprocess

NIR, RED, GREEN, BLUE = 40, 30, 20, 10


def write_four_band(path):
    """Write a 2x2 image in the on-disk layout: NIR, Red, Green, Blue."""
    stored = np.full((2, 2, 4), (NIR, RED, GREEN, BLUE), dtype=np.uint8)
    Image.fromarray(stored).save(path, format="PNG")


def test_opencv_alone_scrambles_four_band_files(tmp_path):
    # Why read_image() exists: OpenCV treats the 4 bands as RGBA and swaps the
    # first three, so plain cv2.imread returns Green, Red, NIR, Blue.
    path = tmp_path / "four_band.tiff"
    write_four_band(path)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    np.testing.assert_array_equal(image[0, 0], [GREEN, RED, NIR, BLUE])


@pytest.mark.parametrize("name", ["four_band.tiff", "four_band.png"])
def test_read_image_returns_band_order(tmp_path, name):
    path = tmp_path / name
    write_four_band(path)

    image = preprocess.read_image(str(path))

    assert preprocess.BAND_ORDER == ["blue", "green", "red", "nir"]
    np.testing.assert_array_equal(image[0, 0], [BLUE, GREEN, RED, NIR])


def test_read_image_leaves_rgb_files_as_bgr(tmp_path):
    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((2, 2, 3), (RED, GREEN, BLUE), dtype=np.uint8)).save(path)

    image = preprocess.read_image(str(path))

    np.testing.assert_array_equal(image[0, 0], [BLUE, GREEN, RED])


def test_gcs_stream_matches_local_read(tmp_path):
    path = tmp_path / "four_band.tiff"
    write_four_band(path)

    streamed = preprocess.stream_image_from_gcs(path.read_bytes())

    np.testing.assert_array_equal(streamed, preprocess.read_image(str(path)))


def test_read_image_returns_none_for_unreadable_file(tmp_path):
    path = tmp_path / "broken.tiff"
    path.write_bytes(b"not an image")

    assert preprocess.read_image(str(path)) is None


def test_enhanced_red_blends_nir_into_red(tmp_path):
    # The fusion must mix the real Red and NIR bands, not Blue and NIR.
    path = tmp_path / "four_band.tiff"
    write_four_band(path)

    fused = preprocess.rgb_nir_fusion(preprocess.read_image(str(path)),
                                      technique="enhanced_red", alpha=0.5)

    np.testing.assert_array_equal(fused[0, 0], [BLUE, GREEN, (RED + NIR) // 2])
