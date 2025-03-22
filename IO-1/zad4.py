from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def simulate_dvb_transmission(image_path):
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
    # Downsampling by factor of 2 in both dimensions (keeping every other pixel)
    # This simulates the chroma subsampling in DVB
    height, width = Y.shape
    Cb_downsampled = Cb[::2, ::2]
    Cr_downsampled = Cr[::2, ::2]

    # Step 3: Perform upsampling on Cb and Cr channels
    # Simple nearest-neighbor upsampling
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

    # Convert back to 0-255 range for display
    rgb_transmitted_display = (rgb_transmitted * 255).astype(np.uint8)

    # Display the results - updated to include all required views
    fig = plt.figure(figsize=(15, 10))

    # Original and transmitted images side by side
    plt.subplot(231)
    plt.imshow(img_array)
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(rgb_transmitted_display)
    plt.title('After DVB Transmission')
    plt.axis('off')

    # Original YCbCr components in grayscale
    plt.subplot(232)
    plt.imshow(Y, cmap='gray')
    plt.title('Original Y Component')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(Y, cmap='gray')  # Y is unchanged in transmission
    plt.title('Transmitted Y Component')
    plt.axis('off')

    # Original and transmitted Cb in grayscale
    plt.subplot(233)
    plt.imshow(Cb, cmap='gray')
    plt.title('Original Cb Component')
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(Cb_upsampled, cmap='gray')
    plt.title('Transmitted Cb Component')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Show another figure for Cr component
    fig2 = plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(Cr, cmap='gray')
    plt.title('Original Cr Component')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(Cr_upsampled, cmap='gray')
    plt.title('Transmitted Cr Component')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return {
        'Y': Y,
        'original_Cb': Cb,
        'original_Cr': Cr,
        'downsampled_Cb': Cb_downsampled,
        'downsampled_Cr': Cr_downsampled,
        'upsampled_Cb': Cb_upsampled,
        'upsampled_Cr': Cr_upsampled,
        'transmitted_image': rgb_transmitted
    }


if __name__ == "__main__":
    image_path = input("Enter image path (e.g., cat.jpg): ")
    result = simulate_dvb_transmission(image_path)
    print("DVB transmission simulation complete!")