"""Placeholder to export captured images to for downlinking
Priortize predictability of size / lossless options

Two current options, export to lossless avif or creating a custom algo to have direct memory allocation control.

AVIF
https://github.com/AOMediaCodec/libavif
libavif and avifenc
avifenc --lossless --speed <0-10> -p --yuv 444 -d 8 input.raw output.avif

CCDS 122.0-B-1 Image Data Compression Standard
Discrete Wavelet Transform (3 level 2D decompostion, produces 10 subands)
Original Image → Apply filter → Get 4 quadrants:
┌─────┬─────┐
│ LL  │ HL  │  LL = Low-Low (blurry version)
├─────┼─────┤  HL = High-Low (vertical edges)
│ LH  │ HH  │  LH = Low-High (horizontal edges)
└─────┴─────┘  HH = High-High (diagonal details)

and Bit Plane Encoder (progressive encoding with exact rate control)

available_space = 64_000
for bit_plane in [7, 6, 5, 4, 3, 2, 1, 0]:
    for subband in [LL3, HL3, LH3, ...]:  #
        if space_left():
            encode_bits(subband, bit_plane)
        else:
            stop()

At high level when we move to static language

struct DownlinkPacket {
    uint16_t tile_id;
    uint16_t quality_used;
    uint32_t actual_size;
    uint8_t data[MAX_TILE_SIZE];
};
"""
import subprocess
import os
from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

import pywt


class AVIFConfig(BaseModel):
    """Configuration for AVIF encoding."""
    speed: int = Field(default=6, ge=0, le=10, description="Encoding speed (0=slowest, 10=fastest)")
    lossless: bool = Field(default=True, description="Enable lossless compression")
    yuv_format: str = Field(default="444", description="YUV subsampling format")
    bit_depth: int = Field(default=8, description="Bit depth")


class CCDSConfig(BaseModel):
    """Configuration for CCDS compression."""
    max_size: int = Field(default=64000, description="Maximum compressed size in bytes")
    wavelet_levels: int = Field(default=3, description="Number of wavelet decomposition levels")
    bit_planes: int = Field(default=8, description="Number of bit planes to encode")

def export_avif(input_image: str, config: Optional[AVIFConfig] = None) -> bool:
    """Export an image to AVIF format using avifenc.

    Args:
        input_image: Path to the input image file
        config: Optional AVIF encoding configuration

    Returns:
        bool: True if encoding succeeded, False otherwise
    """
    if config is None:
        config = AVIFConfig()

    base_name = os.path.splitext(input_image)[0]
    output_image = base_name + '.avif'

    cmd = ['avifenc']
    if config.lossless:
        cmd.append('--lossless')
    cmd.extend(['--speed', str(config.speed), '-p', '--yuv', config.yuv_format,
                '-d', str(config.bit_depth), input_image, output_image])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

# TODO: Implement!
def perform_dwt(image_data: NDArray[np.uint8], levels: int) -> List[NDArray]:
    """Perform multi-level 2D Discrete Wavelet Transform.

    Args:
        image_data: Input image data
        levels: Number of decomposition levels

    Returns:
        List of wavelet subbands (LL, HL, LH, HH for each level)
    """
    coeffs = pywt.wavedec2(image_data, wavelet='haar', level=levels)  
    # coeffs[0] = LLn
    # coeffs[1:] = (HLn, LHn, HHn), (HLn-1, LHn-1, HHn-1), ...
    
    subbands = []
    LLn = coeffs[0]
    subbands.append(LLn)
    for detail_level in coeffs[1:]:
        HL, LH, HH = detail_level
        subbands.extend([HL, LH, HH])
    return subbands

# TODO: Implement!
def encode_bit_plane(subband: NDArray, bit_plane: int) -> bytearray:
    """Encode a specific bit plane of a wavelet subband.

    Args:
        subband: Wavelet subband data
        bit_plane: Bit plane index (0-7)

    Returns:
        Encoded bit plane data
    """
    # Convert to integers
    arr = np.rint(subband).astype(np.int32)  # round in case of floats
    
    # Get the bit mask
    mask = 1 << bit_plane
    bits = (np.abs(arr) & mask) >> bit_plane  # 0 or 1

    # Flatten row-major
    flat_bits = bits.flatten()

    # Pack bits into bytes
    packed = bytearray()
    for i in range(0, len(flat_bits), 8):
        byte_val = 0
        for b in flat_bits[i:i+8]:
            byte_val = (byte_val << 1) | int(b)
        # Pad last byte with zeros if needed
        if len(flat_bits[i:i+8]) < 8:
            byte_val <<= (8 - len(flat_bits[i:i+8]))
        packed.append(byte_val)

    return packed


def test_avif_export():
    from pathlib import Path

    # Path to your test image (make sure it exists)
    input_image = "image.NEF"  # or .jpg, .bmp, etc.
    
    # Check file exists first
    if not Path(input_image).exists():
        print("❌ Input image not found. Please place a test image named 'test_image.png' in this folder.")
        return
    
    # Optional custom config
    config = AVIFConfig(
        speed=4,
        lossless=True,
        yuv_format="444",
        bit_depth=8
    )

    print("🔄 Compressing image to AVIF...")
    success = export_avif(input_image, config)

    if success:
        output_image = Path(input_image).with_suffix('.avif')
        print(f"✅ AVIF export succeeded! Output file: {output_image}")
    else:
        print("❌ AVIF export failed.")
        

def ccds_compression(image_data: NDArray[np.uint8], config: Optional[CCDSConfig] = None) -> bytearray:
    """Compress image data using CCDS 122.0-B-1 standard.

    Implements Discrete Wavelet Transform with Bit Plane Encoding
    for space-constrained image compression.

    Args:
        image_data: Input image as numpy array
        config: Optional CCDS compression configuration

    Returns:
        bytearray: Compressed image data within size limit
    """
    if config is None:
        config = CCDSConfig()

    available_space = config.max_size
    compressed_data = bytearray()

    wavelet_subbands = perform_dwt(image_data, levels=config.wavelet_levels)

    for bit_plane in range(config.bit_planes - 1, -1, -1):
        for subband in wavelet_subbands:
            if len(compressed_data) < available_space:
                encoded_bits = encode_bit_plane(subband, bit_plane)
                compressed_data.extend(encoded_bits)
            else:
                break

    return compressed_data[:available_space]

import rawpy
import numpy as np
from PIL import Image
def test_ccds():
    # Load NEF and convert to RGB
    with rawpy.imread("image.NEF") as raw:
        rgb = raw.postprocess()

    # Convert to grayscale for CCDS test (CCDS spec is often single-band)
    gray = np.mean(rgb, axis=2).astype(np.uint8)

    # Optional: save grayscale preview
    Image.fromarray(gray).save("preview.png")

    # Run CCDS compression
    compressed = ccds_compression(gray)

    print(f"Original size: {gray.size} bytes")
    print(f"Compressed size: {len(compressed)} bytes")
    print(f"First 64 bytes of compressed data: {compressed[:64]}")

test_ccds()

