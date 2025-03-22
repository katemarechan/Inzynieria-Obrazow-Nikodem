
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def transform_colors(image_path):
    # Read the image
    img = Image.open(image_path)

    # Detect if image is color or grayscale
    is_color = img.mode == 'RGB' or img.mode == 'RGBA'

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize values to 0-1 range
    img_norm = img_array.astype(float) / 255.0

    # Define the transformation matrix
#    transformation_matrix = np.array([
#        [0.393, 0.769, 0.189],
#        [0.349, 0.686, 0.168],
#        [0.272, 0.534, 0.131]
#    ])

    transformation_matrix = np.array([
        [0.840, 0.370, 0.160],  # Strong red output
        [0.120, 0.530, 0.140],  # Reduced green
        [0.180, 0.270, 0.350]  # Moderate blue for wine-like depth
    ])

    # Apply transformation
    # Reshape for matrix multiplication
    height, width, channels = img_norm.shape
    pixels = img_norm.reshape((height * width, channels))

    # Apply transformation to each pixel
    transformed_pixels = np.dot(pixels, transformation_matrix.T)

    # Clip values to ensure they're in the 0-1 range
    transformed_pixels = np.clip(transformed_pixels, 0.0, 1.0)

    # Reshape back to image format
    transformed_img = transformed_pixels.reshape((height, width, channels))

    # Convert back to 0-255 range for display
    transformed_img_display = (transformed_img * 255).astype(np.uint8)

    # Display the original and transformed images
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(img_array)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(transformed_img_display)
    plt.title('Transformed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return transformed_img, transformed_img_display

# Example usage

if __name__ == "__main__":
    image_path = input("Enter image path (e.g., cat.jpg): ")
    transformed_img, transformed_img_display = transform_colors(image_path)
    print(f"Transformation complete!")