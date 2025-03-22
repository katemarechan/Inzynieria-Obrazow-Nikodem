from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def rgb_to_ycbcr_conversion(image_path):
    # Read the image
    img = Image.open(image_path)

    # Ensure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert image to numpy array
    img_array = np.array(img)

    # Normalize RGB values to 0-1 range
    rgb_norm = img_array.astype(float) / 255.0

    # Get individual RGB channels
    R = rgb_norm[:, :, 0]
    G = rgb_norm[:, :, 1]
    B = rgb_norm[:, :, 2]

    # Define the YCbCr conversion matrix from the task
    # Y  = 0.229*R + 0.587*G + 0.114*B + 0
    # Cr = 0.500*R - 0.418*G - 0.082*B + 128
    # Cb = -0.168*R - 0.331*G + 0.500*B + 128

    # Calculate YCbCr channels
    Y = 0.229 * R + 0.587 * G + 0.114 * B
    Cr = 0.500 * R - 0.418 * G - 0.082 * B + 128 / 255.0  # Normalize the offset
    Cb = -0.168 * R - 0.331 * G + 0.500 * B + 128 / 255.0  # Normalize the offset

    # Combine into YCbCr image (just for storage)
    ycbcr = np.stack((Y, Cb, Cr), axis=2)

    # Convert back to RGB
    # Define inverse matrix for YCbCr to RGB conversion
    # R = Y + 1.402*(Cr-128/255)
    # G = Y - 0.344*(Cb-128/255) - 0.714*(Cr-128/255)
    # B = Y + 1.772*(Cb-128/255)

    R_conv = Y + 1.402 * (Cr - 128 / 255.0)
    G_conv = Y - 0.344 * (Cb - 128 / 255.0) - 0.714 * (Cr - 128 / 255.0)
    B_conv = Y + 1.772 * (Cb - 128 / 255.0)

    # Clip values to ensure they're in the 0-1 range
    R_conv = np.clip(R_conv, 0.0, 1.0)
    G_conv = np.clip(G_conv, 0.0, 1.0)
    B_conv = np.clip(B_conv, 0.0, 1.0)

    # Combine back to RGB
    rgb_reconstructed = np.stack((R_conv, G_conv, B_conv), axis=2)

    # Convert back to 0-255 range for display
    rgb_reconstructed_display = (rgb_reconstructed * 255).astype(np.uint8)

    # Display the original and components
    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(img_array)
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(Y, cmap='gray')
    plt.title('Y Component (Luminance)')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(Cb, cmap='gray')
    plt.title('Cb Component (Blue Chrominance)')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(Cr, cmap='gray')
    plt.title('Cr Component (Red Chrominance)')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(rgb_reconstructed_display)
    plt.title('Reconstructed RGB Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return ycbcr, rgb_reconstructed

# Example usage

if __name__ == "__main__":
    image_path = input("Enter image path (e.g., cat.jpg): ")
    ycbcr, rgb_reconstructed = rgb_to_ycbcr_conversion(image_path)
    print("YCbCr conversion complete!")