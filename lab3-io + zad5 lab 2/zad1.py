#zad1

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.fftpack import idct


# Quantization matrix as defined in JPEG standard (scaled by 0.5 as in the original code)
QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 48, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float64)
QY = np.ceil(QY / 2)

def dct2(array):
    """Discrete cosine transform."""
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(array):
    """Inverse discrete cosine transform."""
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')

def split_channel_to_blocks(channel):
    """Splits channel into blocks 8x8"""
    blocks = []
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            # Handle edge cases where image dimensions aren't multiples of 8
            block = np.zeros((8, 8), dtype=channel.dtype)
            h = min(8, channel.shape[0] - i)
            w = min(8, channel.shape[1] - j)
            block[:h, :w] = channel[i:i + h, j:j + w]
            blocks.append(block)
    return blocks

def merge_blocks_to_channel(blocks, width, height):
    """Merge 8x8 blocks into a channel"""
    step = int(np.ceil(width / 8))
    rows = []
    for i in range(0, len(blocks), step):
        rows.append(np.concatenate(blocks[i:i + step], axis=1)[:, :width])
    channel = np.concatenate(rows, axis=0)[:height, :]
    return channel

def encode_as_binary_array(message):
    """Convert a string message to a binary string"""
    binary = ""
    for char in message:
        # Convert each character to its 8-bit binary representation
        binary += format(ord(char), '08b')
    return binary

def decode_from_binary_array(binary_string):
    """Convert a binary string to a string message"""
    message = ""
    # Process in chunks of 8 bits
    for i in range(0, len(binary_string), 8):
        if i + 8 <= len(binary_string):  # Make sure we have a full byte
            byte = binary_string[i:i+8]
            message += chr(int(byte, 2))
    return message


def hide_message(blocks, message):
    """Hide a message in DCT blocks by modifying the LSB of non-zero/one coefficients."""
    blocks = [b.copy().astype(np.int32) for b in blocks]  # Make copies to avoid modifying originals
    i = 0
    for nb in range(len(blocks)):
        for x, y in [(x, y) for x in range(8) for y in range(8)]:
            if i >= len(message):
                break
            value = blocks[nb][x, y]
            # Skip zeros and ones as they can cause issues
            if value == 0 or value == 1 or value == -1:
                continue

            m = message[i]
            i += 1

            # Preserve the sign
            sign = 1 if value >= 0 else -1
            value_abs = abs(value)

            # Set the LSB
            if m == '1':
                value_abs = value_abs | 1  # Set LSB to 1
            else:
                value_abs = value_abs & ~1  # Set LSB to 0

            blocks[nb][x, y] = value_abs * sign

    if i < len(message):
        print(f"Warning: Could only encode {i}/{len(message)} bits of the message")
    else:
        print(f"Successfully encoded all {len(message)} bits")
    return blocks


def reveal_message(blocks, length=0):
    """Reveal message from blocks.
    length: length of the message in bits
    """
    blocks = [b.copy().astype(np.int32) for b in blocks]
    message = ""
    i = 0
    for block in blocks:
        for x, y in [(x, y) for x in range(8) for y in range(8)]:
            if i >= length and length > 0:
                return message
            value = block[x, y]
            if value == 0 or value == 1 or value == -1:
                continue

            # Extract the LSB
            lsb = '1' if (abs(value) & 1) == 1 else '0'
            message += lsb
            i += 1
    return message

def y_to_dct_blocks(Y):
    """Convert Y to quantized dct blocks."""
    Y = Y.astype(np.float32)
    blocks = split_channel_to_blocks(Y)
    blocks = [dct2(block) for block in blocks]
    blocks = [np.round(block / QY) for block in blocks]
    return blocks

def dct_blocks_to_y(blocks, image_width, image_height):
    """Convert quantized dct blocks to Y."""
    blocks = [block * QY for block in blocks]
    blocks = [idct2(block) for block in blocks]
    Y = merge_blocks_to_channel(blocks, image_width, image_height).round()
    return Y

def load_image(image_path, rgb=True):
    """Load an image file"""
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    if rgb:
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def save_image(image_path, image):
    """Save an image to a file"""
    if image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
        # For JPEG, convert to BGR color space
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_to_save = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        else:
            image_to_save = image
        cv.imwrite(image_path, image_to_save, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    else:
        # For PNG and other formats
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_to_save = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        else:
            image_to_save = image
        cv.imwrite(image_path, image_to_save)
    print(f"Image saved to {image_path}")

def embed_message_in_image(input_image_path, output_image_path, message, save_jpeg=True):
    """Embed a message in an image and save it"""
    # Load the original image
    original_image = load_image(input_image_path, True)

    # Print original image dimensions
    print(f"Original image dimensions: {original_image.shape}")

    # Make sure dimensions are multiples of 8 (for proper 8x8 DCT blocks)
    height, width = original_image.shape[:2]

    # Convert to YCrCb
    image = cv.cvtColor(original_image, cv.COLOR_RGB2YCrCb)

    # Split channels
    Y = image[:, :, 0]
    Cr = image[:, :, 1]
    Cb = image[:, :, 2]

    # Convert Y to quantized DCT blocks
    blocks = y_to_dct_blocks(Y)

    # Convert message to binary
    binary_message = encode_as_binary_array(message)
    print(f"Message length in bits: {len(binary_message)}")

    # Hide message in DCT blocks
    blocks = hide_message(blocks, binary_message)

    # Verify message is hidden correctly
    message_from_dct = reveal_message(blocks, len(binary_message))
    decoded_message = decode_from_binary_array(message_from_dct)
    print(f"Message verified in DCT domain: {decoded_message}")

    # Convert DCT blocks back to Y channel
    Y_modified = dct_blocks_to_y(blocks, width, height)

    # Merge channels
    image_with_message = np.stack([Y_modified, Cr, Cb], axis=2)

    # Clip values to valid range
    image_with_message = np.clip(image_with_message, 0, 255)
    image_with_message = image_with_message.astype(np.uint8)

    # Convert back to RGB
    image_with_message_rgb = cv.cvtColor(image_with_message, cv.COLOR_YCrCb2RGB)

    # Save the image with the hidden message
    save_image(output_image_path, image_with_message_rgb)

    return original_image, image_with_message_rgb

def extract_message_from_image(image_path, message_length_bits):
    """Extract a hidden message from an image"""
    # Load the image with the hidden message
    loaded_image = load_image(image_path, True)

    # Convert to YCrCb
    loaded_image_ycrcb = cv.cvtColor(loaded_image, cv.COLOR_RGB2YCrCb)

    # Get Y channel
    Y = loaded_image_ycrcb[:, :, 0]

    # Convert Y to DCT blocks
    blocks = y_to_dct_blocks(Y)

    # Reveal the message
    binary_message = reveal_message(blocks, message_length_bits)

    # Convert binary message to text
    decoded_message = decode_from_binary_array(binary_message)

    return decoded_message, loaded_image

def main():
    # Show menu options
    while True:
        print("\n===== JPEG LSB Steganography =====")
        print("1. Hide a message in an image")
        print("2. Extract a message from an image")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            # Get user input for hiding a message
            input_image = input("Enter the path to the input image: ")
            output_image = input("Enter the path for the output image: ")
            message = input("Enter the message to hide: ")

            try:
                print("\nProcessing...")
                original, embedded = embed_message_in_image(input_image, output_image, message)

                # Calculate message length in bits (for future extraction)
                message_length_bits = len(encode_as_binary_array(message))
                print(f"\nMessage successfully hidden!")
                print(f"Remember this number for extraction: {message_length_bits} bits")

                # Display images
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(original)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(embedded)
                plt.title("Image with Hidden Message")
                plt.axis('off')

                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"Error: {e}")

        elif choice == '2':
            # Get user input for extracting a message
            image_path = input("Enter the path to the image with hidden message: ")
            try:
                message_length_bits = int(input("Enter the message length in bits: "))

                print("\nExtracting message...")
                extracted_message, loaded_image = extract_message_from_image(image_path, message_length_bits)

                print(f"\nExtracted message: {extracted_message}")

                # Display the image
                plt.figure(figsize=(8, 8))
                plt.imshow(loaded_image)
                plt.title("Image with Hidden Message")
                plt.axis('off')
                plt.show()

            except ValueError:
                print("Error: Please enter a valid number for message length.")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
