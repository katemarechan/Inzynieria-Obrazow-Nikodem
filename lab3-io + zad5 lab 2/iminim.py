# iminim.py - Image in Image

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def load_image(image_path):
    """Load an image file"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib


def save_image(image_path, image):
    """Save an image to a file"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    cv2.imwrite(image_path, image_to_save)
    print(f"Image saved to {image_path}")


def generate_unique_filename(base_name="stego_image", extension=".png"):
    """Generate a unique filename that doesn't overwrite existing files"""
    if not os.path.exists(f"{base_name}{extension}"):
        return f"{base_name}{extension}"

    counter = 1
    while os.path.exists(f"{base_name}{counter}{extension}"):
        counter += 1

    return f"{base_name}{counter}{extension}"


def hide_image_in_image(cover_image_path, secret_image_path, output_image_path, nbits=1, max_secret_size=None):

    # Load the cover image
    cover_image = load_image(cover_image_path)
    print(f"Cover image loaded: {cover_image.shape}")

    # Load the secret image
    secret_image = load_image(secret_image_path)
    print(f"Secret image loaded: {secret_image.shape}")

    # Calculate available space in the cover image
    if len(cover_image.shape) == 3:  # Color image
        height, width, channels = cover_image.shape
        max_bits = height * width * channels * nbits
    else:  # Grayscale image
        height, width = cover_image.shape
        max_bits = height * width * nbits

    print(f"Available space in cover image: {max_bits} bits")

    # Resize the secret image if necessary
    if max_secret_size:
        secret_image = cv2.resize(secret_image, max_secret_size)
        print(f"Resized secret image to: {secret_image.shape}")

    # Convert secret image to JPEG format with quality compression
    # This step helps reduce the size while maintaining visual quality
    temp_path = "temp_secret.jpg"
    # Convert RGB to BGR for OpenCV
    secret_bgr = cv2.cvtColor(secret_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, secret_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

    # Read the compressed secret image as binary data
    with open(temp_path, "rb") as file:
        secret_data = file.read()

    print(f"Compressed secret image size: {len(secret_data)} bytes")

    # Convert secret image bytes to binary string
    binary_data = ""
    for byte in secret_data:
        binary_data += format(byte, '08b')

    print(f"Binary data length: {len(binary_data)} bits")

    # Check if the cover image is large enough
    if len(binary_data) > max_bits:
        raise ValueError(
            f"Secret image too large to hide. Need {len(binary_data)} bits, but only have {max_bits} bits available.")

    # Create a copy of the cover image
    stego_image = cover_image.copy()

    # Flatten the image
    flat_image = stego_image.reshape(-1)

    # Calculate mask for clearing the LSBs
    mask = ~((1 << nbits) - 1) & 0xFF  # Ensure mask stays in uint8 range

    # Embed the secret image data
    for i in range(0, len(binary_data), nbits):
        if i + nbits <= len(binary_data):
            pixel_index = i // nbits
            if pixel_index < len(flat_image):
                # Clear the LSBs
                flat_image[pixel_index] = (flat_image[pixel_index] & mask).astype(np.uint8)

                # Get nbits from the binary data
                bits = binary_data[i:i + nbits]
                bits_int = int(bits, 2)

                # Set the LSBs
                flat_image[pixel_index] = (flat_image[pixel_index] | bits_int).astype(np.uint8)

    # Remove temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Save the stego image
    save_image(output_image_path, stego_image)

    return len(binary_data)


def main():
    """Main function for image in image steganography"""
    print("=== Image in Image Steganography ===")

    # Get user input for image paths
    cover_image_path = input("Enter the path to the cover image: ")
    secret_image_path = input("Enter the path to the secret image to hide: ")

    # Generate a unique output filename
    output_image_path = generate_unique_filename()
    print(f"Output will be saved as: {output_image_path}")

    try:
        # Ask for image dimensions or use default
        use_defaults = input("Use default settings for image size and encoding bits? (y/n): ").lower()

        if use_defaults == 'y':
            max_secret_size = (150, 150)
            nbits = 3
        else:
            # Get custom dimensions
            width = int(input("Enter maximum width for secret image (e.g., 150): "))
            height = int(input("Enter maximum height for secret image (e.g., 150): "))
            max_secret_size = (width, height)

            # Get bit depth
            nbits = int(input("Enter number of LSB bits to use (1-4 recommended): "))

        # Hide the second image in the first with resizing
        data_length = hide_image_in_image(
            cover_image_path,
            secret_image_path,
            output_image_path,
            nbits=nbits,
            max_secret_size=max_secret_size
        )

        print(f"Successfully hidden {data_length} bits of image data")
        print(f"To extract the hidden image, remember this data length: {data_length}")
        print(f"Or use the automatic extraction method that looks for JPEG markers")

        # Ask if user wants to display the images
        show_images = input("Display the original and steganography images? (y/n): ").lower()
        if show_images == 'y':
            # Load and display images
            cover_image = load_image(cover_image_path)
            stego_image = load_image(output_image_path)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(cover_image)
            plt.title("Original Cover Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(stego_image)
            plt.title("Image with Hidden Content")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error: {e}")

    input("\nPress Enter to return to the main menu...")
