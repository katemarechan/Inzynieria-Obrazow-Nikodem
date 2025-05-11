#zad2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def find_closest_palette_color(value, k=2):
    """
    Find the closest palette color for the given value based on k levels.
    Formula: round((k - 1) * value / 255) * 255 / (k - 1)
    """
    if k == 1:
        return 0  # With only one level, it's always 0

    return round((k - 1) * value / 255) * 255 / (k - 1)


def apply_color_reduction(image_path, output_path=None, k=2):
    """
    Apply just the color reduction without dithering.
    """
    # Open the image in RGB mode
    img = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    img_array = np.array(img, dtype=np.float64)
    height, width, channels = img_array.shape

    # Create a copy to store the result
    result = np.copy(img_array)

    # Apply color reduction
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                result[y, x, c] = find_closest_palette_color(result[y, x, c], k)

    # Convert back to uint8 for saving
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Create a new image from the result array
    reduced_img = Image.fromarray(result)

    # If output path is not specified, create one based on the input path
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_reduced_k{k}{ext}"

    # Save the reduced image
    reduced_img.save(output_path)
    print(f"Color-reduced image saved as {output_path}")

    return reduced_img


def apply_floyd_steinberg_dithering(image_path, output_path=None, k=2):
    """
    Apply Floyd-Steinberg dithering to a color image.
    k determines the number of levels per channel
    """
    # Open the image in RGB mode
    img = Image.open(image_path).convert('RGB')

    # Convert image to numpy array for easier manipulation
    img_array = np.array(img, dtype=np.float64)
    height, width, channels = img_array.shape

    # Create a copy to store the dithered result
    result = np.copy(img_array)

    # Apply Floyd-Steinberg dithering
    for y in range(height):
        for x in range(width):
            # Get old pixel values (all channels)
            old_pixel = result[y, x].copy()

            # Find the closest palette color for each channel
            new_pixel = np.array([find_closest_palette_color(p, k) for p in old_pixel])

            # Update the pixel
            result[y, x] = new_pixel

            # Calculate quantization error for each channel
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
    dithered_img = Image.fromarray(result)

    # If output path is not specified, create one based on the input path
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"{name}_dithered_k{k}{ext}"

    # Save the dithered image
    dithered_img.save(output_path)
    print(f"Dithered image saved as {output_path}")

    return dithered_img


def plot_color_histogram(image, title, subplot_pos):
    """
    Plot color histograms for an image
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Create figure for the histograms
    plt.subplot(subplot_pos)
    plt.title(title)

    # Plot histograms for each channel
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        histogram = np.histogram(img_array[:, :, i], bins=256, range=(0, 255))[0]
        plt.plot(histogram, color=color, alpha=0.5)

    plt.xlim([0, 256])


def main():
    print("Floyd-Steinberg Color Dithering Algorithm (Task 2)")
    print("------------------------------------------------")

    # Ask for image path from the user
    image_path = input("Enter the path to the image file: ")

    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        return

    # Ask for output directory (optional)
    output_dir = input("Enter the output directory (leave blank for same as input): ")
    if not output_dir:
        output_dir = os.path.dirname(image_path)
    elif not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    # Ask for k value (number of levels per channel)
    k_input = input("Enter number of levels per channel (2-256, default: 2): ")
    try:
        k = int(k_input) if k_input else 2
        if k < 2:
            print("Value must be at least 2, setting to 2")
            k = 2
        if k > 256:
            print("Value cannot exceed 256, setting to 256")
            k = 256
    except ValueError:
        print("Invalid value, using default k=2")
        k = 2

    # Create output paths
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    reduced_path = os.path.join(output_dir, f"{name}_reduced_k{k}{ext}")
    dithered_path = os.path.join(output_dir, f"{name}_dithered_k{k}{ext}")

    # Process images
    try:
        print(f"\nProcessing with k={k} ({k} levels per RGB channel)...")

        # Open original image
        original_img = Image.open(image_path).convert('RGB')

        # Apply color reduction without dithering
        reduced_img = apply_color_reduction(image_path, reduced_path, k)

        # Apply dithering
        dithered_img = apply_floyd_steinberg_dithering(image_path, dithered_path, k)

        # Create and save the comparison of all three images with histograms
        plt.figure(figsize=(15, 10))

        # Display images
        plt.subplot(231)
        plt.title("Original")
        plt.imshow(np.array(original_img))
        plt.axis('off')

        plt.subplot(232)
        plt.title(f"Color Reduced (k={k})")
        plt.imshow(np.array(reduced_img))
        plt.axis('off')

        plt.subplot(233)
        plt.title(f"Dithered (k={k})")
        plt.imshow(np.array(dithered_img))
        plt.axis('off')

        # Display histograms
        plot_color_histogram(original_img, "Original Histogram", 234)
        plot_color_histogram(reduced_img, "Reduced Histogram", 235)
        plot_color_histogram(dithered_img, "Dithered Histogram", 236)

        plt.tight_layout()

        # Save the comparison figure
        comparison_path = os.path.join(output_dir, f"{name}_comparison_k{k}.png")
        plt.savefig(comparison_path)
        print(f"Comparison with histograms saved as {comparison_path}")

        # Show the plot
        plt.show()

        print("\nProcessing complete!")

    except Exception as e:
        print(f"An error occurred: {e}")

