import subprocess
import os
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

import pywt
import rawpy
import numpy as np
from PIL import Image
from pathlib import Path
import struct

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

class SubbandHeader:
    """Header information for each subband"""
    def __init__(self, shape: Tuple[int, int], min_val: float, max_val: float):
        self.shape = shape
        self.min_val = min_val
        self.max_val = max_val
        
    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        return struct.pack('!HHdd', self.shape[0], self.shape[1], self.min_val, self.max_val)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SubbandHeader':
        """Deserialize header from bytes"""
        h, w, min_val, max_val = struct.unpack('!HHdd', data)
        return cls((h, w), min_val, max_val)
    
    @classmethod
    def size(cls) -> int:
        """Size of header in bytes"""
        return struct.calcsize('!HHdd')

def export_avif(input_image: str, config: Optional[AVIFConfig] = None) -> bool:
    """Export an image to AVIF format using avifenc."""
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

def perform_dwt(image_data: NDArray[np.uint8], levels: int) -> List[NDArray]:
    """Perform multi-level 2D Discrete Wavelet Transform"""
    # Convert to float for wavelet transform
    float_data = image_data.astype(np.float64)
    
    # Perform DWT
    coeffs = pywt.wavedec2(float_data, wavelet='haar', level=levels, mode='symmetric')
    
    # Extract subbands in order: LL3, HL3, LH3, HH3, HL2, LH2, HH2, HL1, LH1, HH1
    subbands = []
    
    # Add the LL coefficients
    subbands.append(coeffs[0])
    
    # Add detail coefficients from coarse to fine
    for level_coeffs in coeffs[1:]:
        HL, LH, HH = level_coeffs
        subbands.extend([HL, LH, HH])
    
    return subbands

def quantize_coefficients(coeffs: NDArray) -> Tuple[NDArray[np.uint16], float, float]:
    """Maps floating-point wavelet coefficients to 16-bit unsigned integers"""
    """(prepping for bit-plane encoding)"""
    min_val = float(np.min(coeffs))
    max_val = float(np.max(coeffs))
    
    if abs(max_val - min_val) < 1e-10:  # Essentially constant
        return np.zeros_like(coeffs, dtype=np.uint16), min_val, max_val
    
    # Map to 0-65535 range for 16-bit
    normalized = (coeffs - min_val) / (max_val - min_val)
    quantized = np.round(normalized * 65535).astype(np.uint16)
    
    return quantized, min_val, max_val

def dequantize_coefficients(quantized: NDArray[np.uint16], min_val: float, max_val: float) -> NDArray:
    """Dequantize coefficients back to original range."""
    if abs(max_val - min_val) < 1e-10:
        return np.full_like(quantized, min_val, dtype=np.float64)
    
    # Map back from 0-65535 to original range
    normalized = quantized.astype(np.float64) / 65535.0
    return normalized * (max_val - min_val) + min_val

def encode_bit_plane(coeffs: NDArray[np.uint16], bit_plane: int) -> bytes:
    """Encode a specific bit plane."""
    # Create a mask that has a 1 only in the desired bit position
    bit_mask = 1 << bit_plane
    
    # Use bitwise AND to isolate just that one bit in each value
    # Shift that bit down so it's either 0 or 1
    bits = (coeffs & bit_mask) >> bit_plane
    
    # Pack bits into bytes
    flat_bits = bits.flatten()
    packed = bytearray()
    
    for i in range(0, len(flat_bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(flat_bits):
                byte_val |= (flat_bits[i + j] << (7 - j))
        packed.append(byte_val)
    
    return bytes(packed)

def decode_bit_plane(data: bytes, shape: Tuple[int, int], bit_plane: int) -> NDArray[np.uint16]:
    """Decode a bit plane from packed bytes."""
    h, w = shape
    total_bits = h * w
    
    # Unpack bits
    bits = []
    for byte in data:
        for j in range(8):
            if len(bits) < total_bits:
                bits.append((byte >> (7 - j)) & 1)
    
    # Convert to numpy array and reshape
    bits_array = np.array(bits[:total_bits], dtype=np.uint16)
    
    # Apply bit plane value
    bit_values = bits_array * (1 << bit_plane)
    
    return bit_values.reshape(shape)

def ccds_compression_channel(image_channel: NDArray[np.uint8], config: CCDSConfig) -> Tuple[bytes, List[SubbandHeader]]:
    """Compress a single channel using CCDS."""
    # Perform DWT
    subbands = perform_dwt(image_channel, config.wavelet_levels)
    
    # Quantize each subband
    quantized_subbands = []
    headers = []
    
    for subband in subbands:
        quantized, min_val, max_val = quantize_coefficients(subband)
        quantized_subbands.append(quantized)
        headers.append(SubbandHeader(subband.shape, min_val, max_val))
    
    # Create header section
    header_data = bytearray()
    header_data.extend(struct.pack('!I', len(headers)))  # Number of subbands
    for header in headers:
        header_data.extend(header.to_bytes())
    
    # Progressive encoding
    compressed_data = bytearray(header_data)
    header_size = len(compressed_data)
    
    # Track what we've encoded
    encoded_planes = {}  # (subband_idx, bit_plane) -> True
    
    # Encode from MSB to LSB
    for bit_plane in range(15, -1, -1):  # 16-bit values
        for sb_idx, subband in enumerate(quantized_subbands):
            if len(compressed_data) >= config.max_size:
                break
            
            # Encode this bit plane
            plane_data = encode_bit_plane(subband, bit_plane)
            
            if len(compressed_data) + len(plane_data) <= config.max_size:
                compressed_data.extend(plane_data)
                encoded_planes[(sb_idx, bit_plane)] = True
            else:
                # Can't fit this plane
                break
        
        if len(compressed_data) >= config.max_size:
            break
    
    print(f"  Encoded {len(encoded_planes)} bit planes, header size: {header_size}")
    return bytes(compressed_data), headers, encoded_planes

def ccds_decompression_channel(compressed_data: bytes, config: CCDSConfig) -> NDArray[np.uint8]:
    """Decompress a single channel."""
    # Read header
    offset = 0
    num_subbands = struct.unpack('!I', compressed_data[offset:offset+4])[0]
    offset += 4
    
    headers = []
    for _ in range(num_subbands):
        header_bytes = compressed_data[offset:offset+SubbandHeader.size()]
        headers.append(SubbandHeader.from_bytes(header_bytes))
        offset += SubbandHeader.size()
    
    header_size = offset
    print(f"  Header size: {header_size}, {num_subbands} subbands")
    
    # Initialize subbands
    reconstructed_subbands = []
    for header in headers:
        reconstructed_subbands.append(np.zeros(header.shape, dtype=np.uint16))
    
    # Decode bit planes in same order as encoding
    for bit_plane in range(15, -1, -1):  # 16-bit values
        for sb_idx, header in enumerate(headers):
            if offset >= len(compressed_data):
                break
            
            # Calculate expected size for this bit plane
            h, w = header.shape
            num_bits = h * w
            expected_bytes = (num_bits + 7) // 8
            
            if offset + expected_bytes > len(compressed_data):
                # Not enough data left, stop here
                break
            
            # Decode this bit plane
            plane_data = compressed_data[offset:offset + expected_bytes]
            decoded_plane = decode_bit_plane(plane_data, header.shape, bit_plane)
            
            # Accumulate
            reconstructed_subbands[sb_idx] += decoded_plane
            offset += expected_bytes
        
        if offset >= len(compressed_data):
            break
    
    # Dequantize subbands
    dequantized_subbands = []
    for subband, header in zip(reconstructed_subbands, headers):
        dequant = dequantize_coefficients(subband, header.min_val, header.max_val)
        dequantized_subbands.append(dequant)
    
    # Reconstruct coefficient structure
    coeffs = [dequantized_subbands[0]]  # LL
    idx = 1
    for level in range(config.wavelet_levels):
        HL = dequantized_subbands[idx]
        LH = dequantized_subbands[idx + 1]
        HH = dequantized_subbands[idx + 2]
        coeffs.append((HL, LH, HH))
        idx += 3
    
    # Inverse DWT
    reconstructed = pywt.waverec2(coeffs, wavelet='haar', mode='symmetric')
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed

def rgb_compression(image_rgb: NDArray[np.uint8], config: Optional[CCDSConfig] = None) -> bytes:
    """Compress RGB image."""
    if config is None:
        config = CCDSConfig()

    # Split into channels
    channels = [image_rgb[:,:,i] for i in range(3)]
    
    # Compress each channel
    channel_size = config.max_size // 3
    channel_config = CCDSConfig(
        max_size=channel_size,
        wavelet_levels=config.wavelet_levels,
        bit_planes=config.bit_planes
    )
    
    all_compressed = bytearray()
    
    for i, channel in enumerate(channels):
        print(f"Compressing channel {i}...")
        compressed, headers, encoded_planes = ccds_compression_channel(channel, channel_config)
        
        # Pad to exact channel size for easier splitting during decompression
        channel_data = bytearray(compressed)
        while len(channel_data) < channel_size:
            channel_data.append(0)
        
        all_compressed.extend(channel_data[:channel_size])
        print(f"  Channel {i}: {len(compressed)} bytes")
    
    return bytes(all_compressed)

def rgb_decompression(compressed_data: bytes, image_shape: Tuple[int, int, int], config: CCDSConfig) -> NDArray[np.uint8]:
    """Decompress RGB image."""
    h, w, c = image_shape
    channel_size = config.max_size // 3
    
    channel_config = CCDSConfig(
        max_size=channel_size,
        wavelet_levels=config.wavelet_levels,
        bit_planes=config.bit_planes
    )
    
    # Split into channels
    channels = []
    for i in range(3):
        start = i * channel_size
        end = start + channel_size
        channel_data = compressed_data[start:end]
        
        print(f"Decompressing channel {i}...")
        reconstructed = ccds_decompression_channel(channel_data, channel_config)
        channels.append(reconstructed)
    
    # Stack channels
    rgb_result = np.stack(channels, axis=2)
    return rgb_result

def test_ccds():
    """Test CCDS compression."""
    input_path = Path("raw-images")
    output_path = Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    for nef_file in input_path.glob("*.NEF"):
        print(f"Processing {nef_file.name}...")
        
        # Load and resize for testing
        with rawpy.imread(str(nef_file)) as raw:
            rgb = raw.postprocess()

        # Use a reasonable test size
        test_size = (128, 128)
        rgb_small = np.array(Image.fromarray(rgb).resize(test_size))
        print(f"Test image shape: {rgb_small.shape}")

        # Save original
        original_file = output_path / f"original_{nef_file.stem}.png"
        Image.fromarray(rgb_small).save(original_file)

        # Test compression - slightly more generous settings
        config = CCDSConfig(max_size=15000, wavelet_levels=2, bit_planes=8)
        
        print("Compressing...")
        compressed = rgb_compression(rgb_small, config)
        
        print("Decompressing...")
        decoded_rgb = rgb_decompression(compressed, rgb_small.shape, config)
        
        # Save results
        decoded_file = output_path / f"decoded_{nef_file.stem}.png"
        Image.fromarray(decoded_rgb).save(decoded_file)

        # Calculate quality metrics
        original_bytes = rgb_small.nbytes
        compressed_bytes = len(compressed)
        
        print(f"Results:")
        print(f"  Original: {original_bytes} bytes")
        print(f"  Compressed: {compressed_bytes} bytes") 

test_ccds()