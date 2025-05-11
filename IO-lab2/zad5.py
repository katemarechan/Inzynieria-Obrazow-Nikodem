import numpy as np
from PIL import Image
import os


def create_rgb_spectrum(width=800, height=100):
    """Create the RGB spectrum image: Black → Blue → Cyan → White"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Define colors for spectrum path
    colors = [
        (0, 0, 0),  # Black
        (0, 0, 255),  # Blue
        (0, 255, 255),  # Cyan
        (255, 255, 255)  # White
    ]

    # Calculate segment width
    segment_width = width // (len(colors) - 1)

    # Fill the image with interpolated colors
    for i in range(len(colors) - 1):
        c1 = np.array(colors[i])
        c2 = np.array(colors[i + 1])

        start_x = i * segment_width
        end_x = (i + 1) * segment_width

        for x in range(start_x, end_x):
            # Calculate interpolation ratio
            t = (x - start_x) / segment_width
            # Linear interpolation
            color = (1 - t) * c1 + t * c2
            # Fill column with this color
            img[:, x] = color.astype(np.uint8)

    # Handle any remaining pixels
    if width % segment_width != 0:
        img[:, (len(colors) - 1) * segment_width:] = colors[-1]

    return img


# ------ JPEG ALGORITHM STEPS ------

# Step 0: RGB to YCbCr conversion
def rgb_to_ycbcr(img):
    """Convert RGB image to YCbCr color space (Step 0)"""
    # Ensure input is float for calculation
    img_float = img.astype(np.float32)

    # RGB to YCbCr conversion matrix (BT.601 standard)
    ycbcr = np.zeros_like(img_float)

    # Y (luminance)
    ycbcr[:, :, 0] = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    # Cb (blue-difference chroma)
    ycbcr[:, :, 1] = 128 - 0.168736 * img_float[:, :, 0] - 0.331264 * img_float[:, :, 1] + 0.5 * img_float[:, :, 2]
    # Cr (red-difference chroma)
    ycbcr[:, :, 2] = 128 + 0.5 * img_float[:, :, 0] - 0.418688 * img_float[:, :, 1] - 0.081312 * img_float[:, :, 2]

    return ycbcr


# Step 0 Reverse: YCbCr to RGB conversion
def ycbcr_to_rgb(ycbcr):
    """Convert YCbCr image back to RGB color space (Reverse Step 0)"""
    # Ensure input is float for calculation
    ycbcr_float = ycbcr.astype(np.float32)

    # YCbCr to RGB conversion matrix
    rgb = np.zeros_like(ycbcr_float)

    # R
    rgb[:, :, 0] = ycbcr_float[:, :, 0] + 1.402 * (ycbcr_float[:, :, 2] - 128)
    # G
    rgb[:, :, 1] = ycbcr_float[:, :, 0] - 0.344136 * (ycbcr_float[:, :, 1] - 128) - 0.714136 * (
            ycbcr_float[:, :, 2] - 128)
    # B
    rgb[:, :, 2] = ycbcr_float[:, :, 0] + 1.772 * (ycbcr_float[:, :, 1] - 128)

    # Clip values to valid range
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


# Step 1: Chroma subsampling (4:2:0)
def chroma_subsampling(ycbcr):
    """Apply chroma subsampling to YCbCr image (Step 1)"""
    height, width, _ = ycbcr.shape

    # Create new array with subsampled chroma channels
    subsampled = np.zeros_like(ycbcr)

    # Copy Y channel (no subsampling)
    subsampled[:, :, 0] = ycbcr[:, :, 0]

    # Subsample Cb and Cr channels (4:2:0 - reduce by 2 in both dimensions)
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            # Calculate average for 2x2 block
            y_max = min(y + 2, height)
            x_max = min(x + 2, width)

            # Average for Cb
            cb_avg = np.mean(ycbcr[y:y_max, x:x_max, 1])
            # Average for Cr
            cr_avg = np.mean(ycbcr[y:y_max, x:x_max, 2])

            # Set the same value for the entire 2x2 block
            subsampled[y:y_max, x:x_max, 1] = cb_avg
            subsampled[y:y_max, x:x_max, 2] = cr_avg

    return subsampled, ycbcr


# Step 1 Reverse: Upsample chroma channels
def chroma_upsampling(subsampled, original_shape=None):
    """Upsample chroma channels (Reverse Step 1)"""
    # If original shape not provided, use input shape
    if original_shape is None:
        return subsampled

    # No need to actually do anything here since we kept the original dimensions
    # In a real JPEG implementation, this would reconstruct from a more compact representation
    return subsampled


# Step 2: 8x8 block processing and level shift
def level_shift_and_block_split(ycbcr):
    """Apply level shift (subtract 128) and split into 8x8 blocks (Step 2)"""
    height, width, channels = ycbcr.shape

    # Level shift - subtract 128 from all values (center around 0)
    shifted = ycbcr.astype(np.float32) - 128.0

    # Calculate number of blocks (padding if necessary)
    blocks_h = (height + 7) // 8  # Ceiling division
    blocks_w = (width + 7) // 8

    # Create padded image if necessary
    padded_h = blocks_h * 8
    padded_w = blocks_w * 8

    if padded_h != height or padded_w != width:
        padded = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
        padded[:height, :width, :] = shifted
    else:
        padded = shifted

    # Split into 8x8 blocks
    blocks = []
    for c in range(channels):
        channel_blocks = []
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                block = padded[i:i + 8, j:j + 8, c]
                channel_blocks.append(block)
        blocks.append(channel_blocks)

    return blocks, padded, shifted.shape


# Step 2 Reverse: Combine blocks and level unshift
def combine_blocks_and_unshift(blocks, original_shape):
    """Combine 8x8 blocks and add back 128 (Reverse Step 2)"""
    height, width, channels = original_shape

    # Calculate padded dimensions
    blocks_h = (height + 7) // 8
    blocks_w = (width + 7) // 8
    padded_h = blocks_h * 8
    padded_w = blocks_w * 8

    # Initialize padded image
    reconstructed = np.zeros((padded_h, padded_w, channels), dtype=np.float32)

    # Reconstruct each channel from blocks
    for c in range(channels):
        channel_blocks = blocks[c]
        block_idx = 0
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                if block_idx < len(channel_blocks):
                    reconstructed[i:i + 8, j:j + 8, c] = channel_blocks[block_idx]
                    block_idx += 1

    # Level unshift (add 128)
    reconstructed += 128.0

    # Crop to original size
    reconstructed = reconstructed[:height, :width, :]

    return reconstructed


# Step 3: Discrete Cosine Transform (DCT)
def apply_dct(blocks):
    """Apply DCT to each 8x8 block (Step 3)"""
    dct_blocks = []

    for channel_blocks in blocks:
        dct_channel_blocks = []
        for block in channel_blocks:
            # Apply 2D DCT
            dct_block = np.zeros((8, 8), dtype=np.float32)

            for u in range(8):
                for v in range(8):
                    # DCT coefficients
                    cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                    cv = 1.0 / np.sqrt(2) if v == 0 else 1.0

                    sum_val = 0.0
                    for x in range(8):
                        for y in range(8):
                            sum_val += block[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                                (2 * y + 1) * v * np.pi / 16)

                    dct_block[u, v] = 0.25 * cu * cv * sum_val

            dct_channel_blocks.append(dct_block)

        dct_blocks.append(dct_channel_blocks)

    return dct_blocks


# Step 3 Reverse: Inverse Discrete Cosine Transform (IDCT)
def apply_idct(dct_blocks):
    """Apply IDCT to each 8x8 block (Reverse Step 3)"""
    idct_blocks = []

    for channel_blocks in dct_blocks:
        idct_channel_blocks = []
        for dct_block in channel_blocks:
            # Apply 2D IDCT
            idct_block = np.zeros((8, 8), dtype=np.float32)

            for x in range(8):
                for y in range(8):
                    sum_val = 0.0
                    for u in range(8):
                        for v in range(8):
                            # IDCT coefficients
                            cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                            cv = 1.0 / np.sqrt(2) if v == 0 else 1.0

                            sum_val += cu * cv * dct_block[u, v] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                                (2 * y + 1) * v * np.pi / 16)

                    idct_block[x, y] = 0.25 * sum_val

            idct_channel_blocks.append(idct_block)

        idct_blocks.append(idct_channel_blocks)

    return idct_blocks


# Step 4: Quantization
def get_quantization_matrix(quality_factor=50):
    """Generate quantization matrix based on quality factor (Step 4)"""
    # Standard JPEG luminance quantization matrix
    luminance_q_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    # Standard JPEG chrominance quantization matrix
    chrominance_q_matrix = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # Scale matrices based on quality factor
    if quality_factor < 50:
        scale = 5000 / quality_factor
    else:
        scale = 200 - 2 * quality_factor

    scale = scale / 100.0

    # Scale and clip values to ensure they're in range 1-255
    lum_q = np.clip(np.round(luminance_q_matrix * scale), 1, 255)
    chrom_q = np.clip(np.round(chrominance_q_matrix * scale), 1, 255)

    return lum_q, chrom_q


def apply_quantization(dct_blocks, quality_factor=50):
    """Apply quantization to DCT blocks (Step 4)"""
    lum_q_matrix, chrom_q_matrix = get_quantization_matrix(quality_factor)
    quantized_blocks = []

    for c in range(len(dct_blocks)):
        q_matrix = lum_q_matrix if c == 0 else chrom_q_matrix  # Use lum for Y, chrom for Cb/Cr
        quantized_channel_blocks = []

        for block in dct_blocks[c]:
            # Element-wise division by quantization matrix
            quantized_block = block / q_matrix
            quantized_channel_blocks.append(quantized_block)

        quantized_blocks.append(quantized_channel_blocks)

    return quantized_blocks, lum_q_matrix, chrom_q_matrix


# Step 4 Reverse: Dequantization
def apply_dequantization(quantized_blocks, lum_q_matrix, chrom_q_matrix):
    """Apply dequantization to quantized blocks (Reverse Step 4)"""
    dequantized_blocks = []

    for c in range(len(quantized_blocks)):
        q_matrix = lum_q_matrix if c == 0 else chrom_q_matrix  # Use lum for Y, chrom for Cb/Cr
        dequantized_channel_blocks = []

        for block in quantized_blocks[c]:
            # Element-wise multiplication by quantization matrix
            dequantized_block = block * q_matrix
            dequantized_channel_blocks.append(dequantized_block)

        dequantized_blocks.append(dequantized_channel_blocks)

    return dequantized_blocks


# Step 5: Rounding to integers
def apply_rounding(quantized_blocks):
    """Round quantized values to integers (Step 5)"""
    rounded_blocks = []

    for channel_blocks in quantized_blocks:
        rounded_channel_blocks = []

        for block in channel_blocks:
            # Round to nearest integer
            rounded_block = np.round(block)
            rounded_channel_blocks.append(rounded_block)

        rounded_blocks.append(rounded_channel_blocks)

    return rounded_blocks


# Step 5 Reverse: Convert back to float (essentially a pass-through)
def reverse_rounding(rounded_blocks):
    """Convert rounded integers back to float (Reverse Step 5)"""
    # No actual conversion needed since numpy handles this automatically
    # But we'll make a copy to maintain consistency in the pipeline
    float_blocks = []

    for channel_blocks in rounded_blocks:
        float_channel_blocks = []

        for block in channel_blocks:
            float_block = block.astype(np.float32)
            float_channel_blocks.append(float_block)

        float_blocks.append(float_channel_blocks)

    return float_blocks


# Step 6: ZigZag scanning
def get_zigzag_order():
    """Generate ZigZag scan pattern for 8x8 blocks"""
    # Define the ZigZag scan pattern for 8x8 blocks
    zigzag_indices = [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]

    # Convert to (row, col) coordinates
    zigzag_coords = []
    for idx in zigzag_indices:
        row = idx // 8
        col = idx % 8
        zigzag_coords.append((row, col))

    return zigzag_coords


def apply_zigzag(rounded_blocks):
    """Apply ZigZag scanning to convert 2D blocks to 1D sequences (Step 6)"""
    zigzag_coords = get_zigzag_order()
    zigzag_blocks = []

    for channel_blocks in rounded_blocks:
        zigzag_channel_blocks = []

        for block in channel_blocks:
            # Create 1D array following zigzag pattern
            zigzag_sequence = np.zeros(64, dtype=block.dtype)

            for i, (row, col) in enumerate(zigzag_coords):
                zigzag_sequence[i] = block[row, col]

            zigzag_channel_blocks.append(zigzag_sequence)

        zigzag_blocks.append(zigzag_channel_blocks)

    return zigzag_blocks


# Step 6 Reverse: Inverse ZigZag scanning
def apply_inverse_zigzag(zigzag_blocks):
    """Convert 1D ZigZag sequences back to 2D blocks (Reverse Step 6)"""
    zigzag_coords = get_zigzag_order()
    block_size = 8
    inverse_zigzag_blocks = []

    for channel_blocks in zigzag_blocks:
        inverse_zigzag_channel_blocks = []

        for zigzag_sequence in channel_blocks:
            # Create 2D block from 1D zigzag sequence
            block = np.zeros((block_size, block_size), dtype=zigzag_sequence.dtype)

            for i, (row, col) in enumerate(zigzag_coords):
                block[row, col] = zigzag_sequence[i]

            inverse_zigzag_channel_blocks.append(block)

        inverse_zigzag_blocks.append(inverse_zigzag_channel_blocks)

    return inverse_zigzag_blocks


# Step 7: Huffman encoding (simplified version - just calculate size)
def calculate_huffman_size(zigzag_blocks):
    """Calculate approximate size after Huffman encoding (Step 7)"""
    # In a real implementation, this would do actual Huffman encoding
    # For this assignment, we'll just estimate the size

    # Flatten all coefficients
    all_coefficients = []
    for channel_blocks in zigzag_blocks:
        for block in channel_blocks:
            all_coefficients.extend(block.flatten())

    # Count non-zero coefficients (these would be encoded)
    non_zero_count = np.count_nonzero(all_coefficients)

    # Assume average of 4 bits per non-zero coefficient for Huffman coding
    # and 0 bits for zero coefficients (run-length encoded)
    estimated_bits = non_zero_count * 4
    estimated_bytes = estimated_bits / 8

    return estimated_bytes


# Step 7 Reverse: Huffman decoding (not implemented - just pass through)
def huffman_decode(zigzag_blocks):
    """Placeholder for Huffman decoding (Reverse Step 7)"""
    # In a real implementation, this would do actual Huffman decoding
    return zigzag_blocks


# Step 8: Measure final image size
def measure_image_size(zigzag_blocks, original_img):
    """Measure the size of the compressed representation (Step 8)"""
    # Calculate original image size
    original_size = original_img.size * original_img.itemsize

    # Calculate estimated size after compression
    compressed_size = calculate_huffman_size(zigzag_blocks)

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio
    }


# Full JPEG encoding process (Steps 0-8)
def jpeg_encode(img, quality_factor=50):
    """Apply full JPEG encoding steps 0-8"""
    print("Step 0: Converting RGB to YCbCr")
    ycbcr = rgb_to_ycbcr(img)

    print("Step 1: Applying chroma subsampling")
    subsampled, original_ycbcr = chroma_subsampling(ycbcr)

    print("Step 2: Level shifting and splitting into 8x8 blocks")
    blocks, padded, original_shape = level_shift_and_block_split(subsampled)

    print("Step 3: Applying DCT to each block")
    dct_blocks = apply_dct(blocks)

    print("Step 4: Applying quantization with quality factor", quality_factor)
    quantized_blocks, lum_q_matrix, chrom_q_matrix = apply_quantization(dct_blocks, quality_factor)

    print("Step 5: Rounding quantized values to integers")
    rounded_blocks = apply_rounding(quantized_blocks)

    print("Step 6: Applying ZigZag scanning")
    zigzag_blocks = apply_zigzag(rounded_blocks)

    print("Step 7: Applying Huffman encoding (size estimation)")
    estimated_size = calculate_huffman_size(zigzag_blocks)

    print("Step 8: Measuring final image size")
    size_info = measure_image_size(zigzag_blocks, img)

    return {
        'ycbcr': ycbcr,
        'subsampled': subsampled,
        'blocks': blocks,
        'dct_blocks': dct_blocks,
        'quantized_blocks': quantized_blocks,
        'rounded_blocks': rounded_blocks,
        'zigzag_blocks': zigzag_blocks,
        'size_info': size_info,
        'original_shape': original_shape,
        'lum_q_matrix': lum_q_matrix,
        'chrom_q_matrix': chrom_q_matrix
    }


# Full JPEG decoding process (Reverse steps 8-0)
def jpeg_decode(encoded_data):
    """Apply full JPEG decoding steps (reverse 8-0)"""
    print("Reverse Step 7: Huffman decoding")
    zigzag_blocks = huffman_decode(encoded_data['zigzag_blocks'])

    print("Reverse Step 6: Inverse ZigZag scanning")
    inverse_zigzag_blocks = apply_inverse_zigzag(zigzag_blocks)

    print("Reverse Step 5: Converting integers back to float")
    float_blocks = reverse_rounding(inverse_zigzag_blocks)

    print("Reverse Step 4: Applying dequantization")
    dequantized_blocks = apply_dequantization(
        float_blocks,
        encoded_data['lum_q_matrix'],
        encoded_data['chrom_q_matrix']
    )

    print("Reverse Step 3: Applying IDCT to each block")
    idct_blocks = apply_idct(dequantized_blocks)

    print("Reverse Step 2: Combining blocks and level unshifting")
    reconstructed = combine_blocks_and_unshift(idct_blocks, encoded_data['original_shape'])

    print("Reverse Step 1: Upsampling chroma channels")
    upsampled = chroma_upsampling(reconstructed)

    print("Reverse Step 0: Converting YCbCr back to RGB")
    rgb = ycbcr_to_rgb(upsampled)

    return rgb


# Function to compute PSNR (Peak Signal-to-Noise Ratio)
def compute_psnr(original, reconstructed):
    """Compute PSNR between original and reconstructed images"""
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Test different quality factors and evaluate impact
def test_quality_factors(img):
    """Test different quality factors and evaluate their impact on image quality and size"""
    results = {}
    quality_factors = [1, 10, 50, 90]

    for qf in quality_factors:
        print(f"\nTesting quality factor: {qf}")
        encoded_data = jpeg_encode(img, quality_factor=qf)
        reconstructed = jpeg_decode(encoded_data)

        # Calculate PSNR
        psnr = compute_psnr(img, reconstructed)

        # Save reconstructed image
        output_filename = f"reconstructed_qf_{qf}.png"
        Image.fromarray(reconstructed.astype(np.uint8)).save(output_filename)

        # Store results
        results[qf] = {
            'psnr': psnr,
            'size_info': encoded_data['size_info'],
            'reconstructed_image': output_filename
        }

        print(
            f"Quality Factor {qf} - PSNR: {psnr:.2f} dB, Compression Ratio: {encoded_data['size_info']['compression_ratio']:.2f}x")

    return results


def main():
    print("Creating RGB spectrum image...")
    img = create_rgb_spectrum(width=800, height=100)
    Image.fromarray(img).save("original_spectrum.png")

    print("\nTesting different quality factors...")
    qf_results = test_quality_factors(img)

    # Print summary table
    print("\nQuality Factor Impact Summary:")
    print("QF\tPSNR (dB)\tCompression Ratio")
    print("-" * 40)
    for qf in sorted(qf_results.keys()):
        print(f"{qf}\t{qf_results[qf]['psnr']:.2f}\t\t{qf_results[qf]['size_info']['compression_ratio']:.2f}x")

    print("\nObservations:")
    print("1. Lower quality factors (1-10) provide higher compression ratios but lower image quality.")
    print("2. Higher quality factors (90+) preserve image quality but offer less compression.")
    print("3. Quantization (Step 4) has the most significant impact on the compression-quality tradeoff.")
    print("4. The ZigZag scanning (Step 6) enhances the efficiency of Huffman encoding by grouping zeros together.")
    print("5. JPEG is particularly efficient for natural images but may introduce artifacts in sharp transitions.")


if __name__ == "__main__":
    main()