from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def high_pass_filter(image_path):
    # Read the image (PIL handles the image type automatically)
    img = Image.open(image_path)

    # Convert image to numpy array for processing
    img_array = np.array(img)

    # Define the high-pass filter kernel (Sobel operator for better edge detection)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    # Get image dimensions
    if len(img_array.shape) == 3:  # Color image (has channels)
        height, width, channels = img_array.shape
    else:  # Grayscale image
        height, width = img_array.shape
        # Expand dimensions to make processing uniform
        img_array = np.expand_dims(img_array, axis=2)
        channels = 1

    # Create output image array
    filtered_img = np.zeros_like(img_array)

    # Apply the filter using convolution
    for c in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Extract the 3x3 neighborhood around the pixel
                neighborhood = img_array[i - 1:i + 2, j - 1:j + 2, c if channels > 1 else 0]
                # Apply the kernel
                value = np.sum(neighborhood * kernel)
                # Clip values to valid range
                filtered_img[i, j, c if channels > 1 else 0] = np.clip(value, 0, 255)

    # Remove the extra dimension for grayscale images
    if channels == 1:
        filtered_img = filtered_img[:, :, 0]
        img_array = img_array[:, :, 0]

    # Display the original and filtered images
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(img_array, cmap='gray' if channels == 1 else None)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(filtered_img, cmap='gray' if channels == 1 else None)
    plt.title('High-Pass Filtered Image (Edge Detection)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return filtered_img


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    high_pass_filter(image_path)