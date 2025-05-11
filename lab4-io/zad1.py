#zad1

import numpy as np
from PIL import Image
import os


def find_closest_palette_color(value):
    """
    Find the closest palette color for the given value.
    For grayscale, this rounds to either 0 or 255 (black or white).
    """
    return round(value / 255) * 255


def apply_floyd_steinberg_dithering(image_path, output_path=None):
    """
    Apply Floyd-Steinberg dithering to a grayscale image.
    """
    # Open the image and convert to grayscale mode ('L')
    img = Image.open(image_path).convert('L')

    # Convert image to numpy array for easier manipulation
    img_array = np.array(img, dtype=np.float64)
    height, width = img_array.shape

    # Create a copy to store the dithered result
    result = np.copy(img_array)

    # Apply Floyd-Steinberg dithering
    for y in range(height):
        for x in range(width):
            # Get the old pixel value
            old_pixel = result[y, x]
            new_pixel = find_closest_palette_color(old_pixel)

            # Update the pixel
            result[y, x] = new_pixel

            # Calculate quantization error
            quant_error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels
            # Make sure we don't go out of bounds
            if x + 1 < width:
                result[y, x + 1] = np.clip(result[y, x + 1] + quant_error * 7 / 16, 0, 255)

            if y + 1 < height:
                if x - 1 >= 0:
                    result[y + 1, x - 1] = np.clip(result[y + 1, x - 1] + quant_error * 3 / 16, 0, 255)

                result[y + 1, x] = np.clip(result[y + 1, x] + quant_error * 5 / 16, 0, 255)

                if x + 1 < width:
                    result[y + 1, x + 1] = np.clip(result[y + 1, x + 1] + quant_error * 1 / 16, 0, 255)

    # Convert back to uint8 for saving
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Create a new image from the result array
    dithered_img = Image.fromarray(result.astype('uint8'), mode='L')

    # If output path is not specified, create one based on the input path
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_dithered{ext}"

    # Save the dithered image
    dithered_img.save(output_path)
    print(f"Dithered image saved as {output_path}")

    return dithered_img


def main():
    # Ask for image path from the user
    image_path = input("Enter the path to the image file: ")

    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    # Ask for output path (optional)
    output_path = input("Enter the path for the output image (leave blank for default): ")
    if not output_path:
        output_path = None

    # Apply dithering
    try:
        # Check if the image is grayscale
        img = Image.open(image_path)
        if img.mode != 'L':
            print("Converting color image to grayscale...")
            img = img.convert('L')
            # Save the grayscale version temporarily
            temp_path = f"temp_grayscale_{os.path.basename(image_path)}"
            img.save(temp_path)
            image_path = temp_path

        dithered_img = apply_floyd_steinberg_dithering(image_path, output_path)

        # Display original and dithered images
        original_img = Image.open(image_path)

        # You can use the following to display the images,
        # but it requires matplotlib to be installed
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original (Grayscale)")
            plt.imshow(np.array(original_img), cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Dithered")
            plt.imshow(np.array(dithered_img), cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            # Clean up temporary file if created
            if image_path.startswith("temp_grayscale_"):
                os.remove(image_path)

        except ImportError:
            print("Matplotlib not installed. Images won't be displayed.")

            # Clean up temporary file if created
            if image_path.startswith("temp_grayscale_"):
                os.remove(image_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up temporary file if error occurs
        if 'image_path' in locals() and isinstance(image_path, str) and image_path.startswith("temp_grayscale_"):
            if os.path.exists(image_path):
                os.remove(image_path)

