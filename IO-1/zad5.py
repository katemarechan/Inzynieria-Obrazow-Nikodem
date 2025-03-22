from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse(image_path):
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

    # Step 1: Convert RGB to YCbCr
    Y = 0.229 * R + 0.587 * G + 0.114 * B
    Cr = 0.500 * R - 0.418 * G - 0.082 * B + 128 / 255.0
    Cb = -0.168 * R - 0.331 * G + 0.500 * B + 128 / 255.0

    # Step 2: Perform downsampling on Cb and Cr channels
    height, width = Y.shape
    Cb_downsampled = Cb[::2, ::2]
    Cr_downsampled = Cr[::2, ::2]

    # Step 3: Perform upsampling on Cb and Cr channels
    Cb_upsampled = np.zeros_like(Cb)
    Cr_upsampled = np.zeros_like(Cr)

    # Replicate each downsampled pixel to a 2x2 block in the upsampled version
    for i in range(Cb_downsampled.shape[0]):
        for j in range(Cb_downsampled.shape[1]):
            i2 = i * 2
            j2 = j * 2

            # Handle edge cases for odd dimensions
            if i2 < height and j2 < width:
                Cb_upsampled[i2, j2] = Cb_downsampled[i, j]

                if j2 + 1 < width:
                    Cb_upsampled[i2, j2 + 1] = Cb_downsampled[i, j]

                if i2 + 1 < height:
                    Cb_upsampled[i2 + 1, j2] = Cb_downsampled[i, j]

                if i2 + 1 < height and j2 + 1 < width:
                    Cb_upsampled[i2 + 1, j2 + 1] = Cb_downsampled[i, j]

    # Same for Cr
    for i in range(Cr_downsampled.shape[0]):
        for j in range(Cr_downsampled.shape[1]):
            i2 = i * 2
            j2 = j * 2

            if i2 < height and j2 < width:
                Cr_upsampled[i2, j2] = Cr_downsampled[i, j]

                if j2 + 1 < width:
                    Cr_upsampled[i2, j2 + 1] = Cr_downsampled[i, j]

                if i2 + 1 < height:
                    Cr_upsampled[i2 + 1, j2] = Cr_downsampled[i, j]

                if i2 + 1 < height and j2 + 1 < width:
                    Cr_upsampled[i2 + 1, j2 + 1] = Cr_downsampled[i, j]

    # Step 4: Reconstruct RGB from Y and upsampled Cb, Cr
    R_reconstructed = Y + 1.402 * (Cr_upsampled - 128 / 255.0)
    G_reconstructed = Y - 0.344 * (Cb_upsampled - 128 / 255.0) - 0.714 * (Cr_upsampled - 128 / 255.0)
    B_reconstructed = Y + 1.772 * (Cb_upsampled - 128 / 255.0)

    # Clip values to ensure they're in the 0-1 range
    R_reconstructed = np.clip(R_reconstructed, 0.0, 1.0)
    G_reconstructed = np.clip(G_reconstructed, 0.0, 1.0)
    B_reconstructed = np.clip(B_reconstructed, 0.0, 1.0)

    # Combine back to RGB
    rgb_transmitted = np.stack((R_reconstructed, G_reconstructed, B_reconstructed), axis=2)

    # Step 5: Calculate the Mean Square Error (MSE)
    # m = number of channels (3 for RGB)
    # n = number of pixels (height * width)

    # Calculate the squared difference between original and reconstructed image for each channel
    diff_R = (R - R_reconstructed) ** 2
    diff_G = (G - G_reconstructed) ** 2
    diff_B = (B - B_reconstructed) ** 2

    # Calculate MSE according to the formula
    m = 3  # Number of channels (RGB)
    n = height * width  # Number of pixels

    mse = (1 / m) * (1 / n) * (np.sum(diff_R) + np.sum(diff_G) + np.sum(diff_B))

    # Display the results
    # Calculate the difference images
    diff_img_R = np.abs(R - R_reconstructed)
    diff_img_G = np.abs(G - G_reconstructed)
    diff_img_B = np.abs(B - B_reconstructed)

    # Combine for RGB difference visualization
    diff_img = np.stack((diff_img_R, diff_img_G, diff_img_B), axis=2)

    # Enhance the difference for better visibility (multiply by a factor)
    diff_img_enhanced = np.clip(diff_img * 5, 0, 1)

    # Convert to 0-255 range for display
    rgb_original_display = (rgb_norm * 255).astype(np.uint8)
    rgb_transmitted_display = (rgb_transmitted * 255).astype(np.uint8)
    diff_img_display = (diff_img * 255).astype(np.uint8)
    diff_img_enhanced_display = (diff_img_enhanced * 255).astype(np.uint8)

    plt.figure(figsize=(15, 10))

    plt.subplot(221)
    plt.imshow(rgb_original_display)
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(rgb_transmitted_display)
    plt.title('After DVB Transmission')
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(diff_img_display)
    plt.title('Difference Image (Absolute)')
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(diff_img_enhanced_display)
    plt.title(f'Enhanced Difference (MSE = {mse:.8f})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Display MSE for individual color channels
    mse_R = (1 / n) * np.sum(diff_R)
    mse_G = (1 / n) * np.sum(diff_G)
    mse_B = (1 / n) * np.sum(diff_B)

    print(f"MSE for entire image: {mse:.8f}")
    print(f"MSE for R channel: {mse_R:.8f}")
    print(f"MSE for G channel: {mse_G:.8f}")
    print(f"MSE for B channel: {mse_B:.8f}")

    return {
        'mse': mse,
        'mse_R': mse_R,
        'mse_G': mse_G,
        'mse_B': mse_B,
        'original_img': rgb_norm,
        'transmitted_img': rgb_transmitted,
        'difference_img': diff_img,
        'enhanced_difference': diff_img_enhanced
    }


if __name__ == "__main__":
    image_path = input("Enter image path (e.g., cat.jpg): ")
    result = calculate_mse(image_path)
    print(f"MSE calculation complete! Total MSE: {result['mse']:.8f}")