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


# Step 7: Huffman encoding (simplified version - just calculate size)
def calculate_huffman_size(dct_blocks):
    """Calculate approximate size after Huffman encoding (Step 7)"""
    # In a real implementation, this would do actual Huffman encoding
    # For this assignment, we'll just estimate the size

    # Flatten all DCT coefficients
    all_coefficients = []
    for channel_blocks in dct_blocks:
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
def huffman_decode(dct_blocks):
    """Placeholder for Huffman decoding (Reverse Step 7)"""
    # In a real implementation, this would do actual Huffman decoding
    return dct_blocks


# Step 8: Measure final image size
def measure_image_size(dct_blocks, original_img):
    """Measure the size of the compressed representation (Step 8)"""
    # Calculate original image size
    original_size = original_img.size * original_img.itemsize

    # Calculate estimated size after compression
    compressed_size = calculate_huffman_size(dct_blocks)

    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio
    }


# Full JPEG encoding process (Steps 0-3, 7-8)
def jpeg_encode(img):
    """Apply JPEG encoding steps 0, 1, 2, 3, 7, 8"""
    print("Step 0: Converting RGB to YCbCr")
    ycbcr = rgb_to_ycbcr(img)

    print("Step 1: Applying chroma subsampling")
    subsampled, original_ycbcr = chroma_subsampling(ycbcr)

    print("Step 2: Level shifting and splitting into 8x8 blocks")
    blocks, padded, original_shape = level_shift_and_block_split(subsampled)

    print("Step 3: Applying DCT to each block")
    dct_blocks = apply_dct(blocks)

    print("Step 7: Applying Huffman encoding (size estimation)")
    estimated_size = calculate_huffman_size(dct_blocks)

    print("Step 8: Measuring final image size")
    size_info = measure_image_size(dct_blocks, img)

    return {
        'ycbcr': ycbcr,
        'subsampled': subsampled,
        'blocks': blocks,
        'dct_blocks': dct_blocks,
        'size_info': size_info,
        'original_shape': original_shape
    }


# Full JPEG decoding process (Reverse steps 8, 7, 3, 2, 1, 0)
def jpeg_decode(encoded_data):
    """Apply JPEG decoding steps (reverse 8, 7, 3, 2, 1, 0)"""
    print("Reverse Step 7: Huffman decoding")
    dct_blocks = huffman_decode(encoded_data['dct_blocks'])

    print("Reverse Step 3: Applying IDCT to each block")
    idct_blocks = apply_idct(dct_blocks)

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


# Function to evaluate the impact of Step 2 modifications
def test_step2_modifications(encoded_data):
    """Test modifications to Step 2 and evaluate their impact"""
    results = {}

    # Test without every second element
    print("Testing without every second element...")
    blocks_mod = [[] for _ in range(len(encoded_data['blocks']))]
    for c in range(len(encoded_data['blocks'])):
        for block in encoded_data['blocks'][c]:
            mod_block = block.copy()
            mod_block[::2, :] = 0
            blocks_mod[c].append(mod_block)

    dct_mod = apply_dct(blocks_mod)
    idct_mod = apply_idct(dct_mod)
    reconstructed_mod = combine_blocks_and_unshift(idct_mod, encoded_data['original_shape'])
    upsampled_mod = chroma_upsampling(reconstructed_mod)
    rgb_mod = ycbcr_to_rgb(upsampled_mod)

    # Save the modified image
    Image.fromarray(rgb_mod.astype(np.uint8)).save("modified_second_element.png")

    # Test without every fourth element
    print("Testing without every fourth element...")
    blocks_mod2 = [[] for _ in range(len(encoded_data['blocks']))]
    for c in range(len(encoded_data['blocks'])):
        for block in encoded_data['blocks'][c]:
            mod_block = block.copy()
            mod_block[::4, :] = 0
            blocks_mod2[c].append(mod_block)

    dct_mod2 = apply_dct(blocks_mod2)
    idct_mod2 = apply_idct(dct_mod2)
    reconstructed_mod2 = combine_blocks_and_unshift(idct_mod2, encoded_data['original_shape'])
    upsampled_mod2 = chroma_upsampling(reconstructed_mod2)
    rgb_mod2 = ycbcr_to_rgb(upsampled_mod2)

    # Save the modified image
    Image.fromarray(rgb_mod2.astype(np.uint8)).save("modified_fourth_element.png")

    return {
        'second_element_removed': rgb_mod,
        'fourth_element_removed': rgb_mod2
    }


def main():
    print("Creating RGB spectrum image...")
    img = create_rgb_spectrum(width=800, height=100)
    Image.fromarray(img).save("original_spectrum.png")

    print("\nApplying JPEG encoding steps...")
    encoded_data = jpeg_encode(img)

    size_info = encoded_data['size_info']
    print("\nCompression results:")
    print(f"Original size: {size_info['original_size']} bytes")
    print(f"Estimated compressed size: {size_info['compressed_size']:.2f} bytes")
    print(f"Compression ratio: {size_info['compression_ratio']:.2f}x")

    print("\nApplying JPEG decoding steps...")
    reconstructed = jpeg_decode(encoded_data)

    # Save the reconstructed image
    Image.fromarray(reconstructed.astype(np.uint8)).save("reconstructed_spectrum.png")

    # Calculate PSNR
    psnr = compute_psnr(img, reconstructed)
    print(f"\nReconstruction quality (PSNR): {psnr:.2f} dB")

    # Test Step 2 modifications
    print("\nEvaluating impact of Step 2...")
    modified_images = test_step2_modifications(encoded_data)

    # Calculate PSNR for modified images
    psnr_second = compute_psnr(img, modified_images['second_element_removed'])
    psnr_fourth = compute_psnr(img, modified_images['fourth_element_removed'])

    print("\nStep 2 impact assessment:")
    print(f"PSNR with every second element removed: {psnr_second:.2f} dB")
    print(f"PSNR with every fourth element removed: {psnr_fourth:.2f} dB")

    print("\nObservations on the impact of Step 2:")
    print("1. Block division is crucial for spatial localization of frequency components.")
    print("2. Level shifting centers values around 0, improving DCT efficiency.")
    print("3. Removing every second element causes significant quality degradation.")
    print("4. Removing every fourth element has less impact, showing frequency importance.")
    print("5. Step 2 contributes significantly to compression by preparing data for DCT.")


if __name__ == "__main__":
    main()