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
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

# Set up logging
logger = logging.getLogger(__name__)


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
        logger.error(f"Error: {result.stderr}")
        return False
    return True

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

# TODO: Implement!
def perform_dwt(image_data: NDArray[np.uint8], levels: int) -> List[NDArray]:
    """Perform multi-level 2D Discrete Wavelet Transform.

    Args:
        image_data: Input image data
        levels: Number of decomposition levels

    Returns:
        List of wavelet subbands (LL, HL, LH, HH for each level)
    """
    return []

# TODO: Implement!
def encode_bit_plane(subband: NDArray, bit_plane: int) -> bytearray:
    """Encode a specific bit plane of a wavelet subband.

    Args:
        subband: Wavelet subband data
        bit_plane: Bit plane index (0-7)

    Returns:
        Encoded bit plane data
    """
    return bytearray()