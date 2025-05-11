#zad2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os

def encode_as_binary_array(message):
    """Convert a string message to a binary string"""
    return ''.join(format(ord(char), '08b') for char in message)

def decode_from_binary_array(binary_string):
    """Convert a binary string to a string message"""
    message = ""
    # Process in chunks of 8 bits
    for i in range(0, len(binary_string), 8):
        if i + 8 <= len(binary_string):  # Make sure we have a full byte
            byte = binary_string[i:i+8]
            message += chr(int(byte, 2))
    return message

def load_image(image_path):
    """Load an image file"""
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def save_image(image_path, image):
    """Save an image to a file"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_to_save = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    else:
        image_to_save = image

    cv.imwrite(image_path, image_to_save)
    print(f"Image saved to {image_path}")

def hide_message(image, message, nbits=1):
    """
    Hide a binary message in the least significant bits of the image.

    Parameters:
    - image: numpy array containing the image
    - message: binary string (sequence of '0' and '1')
    - nbits: number of least significant bits to use for hiding

    Returns:
    - image with hidden message
    """
    # Make a copy of the image to avoid modifying the original
    result = image.copy()

    # Flatten the image to make it easier to iterate through all pixels
    flat_image = result.reshape(-1)

    # Calculate how many pixels we need to store the message
    required_pixels = len(message) // nbits
    required_pixels += 1 if len(message) % nbits != 0 else 0

    if required_pixels > flat_image.shape[0]:
        raise ValueError(f"Message too long to hide in this image using {nbits} bits")

    # Print percentage of image used
    percentage_used = (required_pixels / flat_image.shape[0]) * 100
    print(f"Using {percentage_used:.2f}% of the image to hide the message with {nbits} bits")

    # Create a bit mask for clearing the LSBs
    mask = 0xFF & (0xFF << nbits)

    # Iterate through the message
    msg_idx = 0
    for px_idx in range(min(required_pixels, flat_image.shape[0])):
        if msg_idx >= len(message):
            break

        # Get current pixel value
        pixel_value = int(flat_image[px_idx])

        # Clear the nbits least significant bits
        pixel_value = pixel_value & mask

        # Calculate how many bits we can set in this pixel
        bits_to_set = min(nbits, len(message) - msg_idx)

        # Get the bits from the message
        msg_bits = message[msg_idx:msg_idx + bits_to_set]

        # Convert the message bits to integer
        msg_value = int(msg_bits, 2)

        # Set the bits
        pixel_value = pixel_value | msg_value

        # Ensure the pixel value is in valid range [0, 255]
        pixel_value = min(255, max(0, pixel_value))

        # Update the pixel
        flat_image[px_idx] = pixel_value

        # Move to the next chunk of the message
        msg_idx += bits_to_set

    if msg_idx < len(message):
        print(f"Warning: Could only encode {msg_idx}/{len(message)} bits of the message")

    return result

def reveal_message(image, nbits=1, length=0):
    """
    Reveal a message hidden in the image.

    Parameters:
    - image: numpy array containing the image with hidden message
    - nbits: number of least significant bits used for hiding
    - length: length of the hidden message in bits (0 for extracting all)

    Returns:
    - binary string containing the hidden message
    """
    # Flatten the image
    flat_image = image.reshape(-1)

    # Calculate how many pixels we need to check
    pixels_to_check = flat_image.shape[0]
    if length > 0:
        required_pixels = length // nbits
        required_pixels += 1 if length % nbits != 0 else 0
        pixels_to_check = min(required_pixels, pixels_to_check)

    # Create bit mask for the LSBs
    mask = (1 << nbits) - 1

    # Extract the message
    message = ""
    for px_idx in range(pixels_to_check):
        # Get the pixel value
        pixel_value = int(flat_image[px_idx])

        # Extract the least significant bits
        lsb_value = pixel_value & mask

        # Convert to binary string and pad with zeros
        bits = format(lsb_value, f'0{nbits}b')
        message += bits

        # Check if we've extracted enough bits
        if length > 0 and len(message) >= length:
            return message[:length]

    return message

def calculate_mse(original, modified):
    """Calculate Mean Square Error between two images"""
    if original.shape != modified.shape:
        raise ValueError("Images must have the same dimensions")

    # Convert to float64 to avoid overflow
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)

    # Calculate MSE
    mse = np.mean((original - modified) ** 2)
    return mse

def generate_lorem_ipsum(paragraphs=10):
    """Generate Lorem Ipsum text with the specified number of paragraphs"""
    lorem_ipsum = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    
    Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor.
    
    Ut ullamcorper, ligula eu tempor congue, eros est euismod turpis, id tincidunt sapien risus a quam. Maecenas fermentum consequat mi. Donec fermentum. Pellentesque malesuada nulla a mi. Duis sapien sem, aliquet nec, commodo eget, consequat quis, neque.
    
    Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Nullam in ipsum. Pellentesque rutrum lorem in dolor. Nullam cursus mi in lectus. Nullam egestas quam a dui. Nunc urna lorem, cursus sit amet, suscipit ac, facilisis sit amet, lorem.
    
    Aliquam egestas wisi eget lorem. Etiam convallis, velit a accumsan rhoncus, dui magna egestas tellus, eu tincidunt quam augue vel turpis. Nam pellentesque odio nec augue. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
    """

    paragraphs_list = lorem_ipsum.strip().split('\n\n')
    # Repeat the paragraphs if more are requested
    while len(paragraphs_list) < paragraphs:
        paragraphs_list.extend(paragraphs_list)

    return ' '.join(paragraphs_list[:paragraphs])

def display_images_with_cv2(images_list, titles):
    """Display images using OpenCV"""
    for i, (image, title) in enumerate(zip(images_list, titles)):
        # Create a named window for each image
        window_name = f"Image {i+1}: {title}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # Convert RGB to BGR for OpenCV display
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_to_display = cv.cvtColor(image.astype(np.uint8), cv.COLOR_RGB2BGR)
        else:
            image_to_display = image.astype(np.uint8)

        # Display the image
        cv.imshow(window_name, image_to_display)

        # Resize window to a reasonable size
        cv.resizeWindow(window_name, 800, 600)

    print("\nDisplaying images. Press any key in image windows to continue...\n")
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    # Ask the user for the image path
    while True:
        input_image_path = input("Please enter the path to the image file: ").strip()
        if os.path.exists(input_image_path):
            break
        else:
            print(f"Error: File '{input_image_path}' does not exist. Please try again.")

    try:
        # Load the original image
        original_image = load_image(input_image_path)
        print(f"Original image dimensions: {original_image.shape}")

        # Generate a long text message (Lorem Ipsum)
        raw_message = generate_lorem_ipsum(paragraphs=30)  # Adjust paragraphs to get a large message
        print(f"Generated message length: {len(raw_message)} characters")

        # Convert to binary
        binary_message = encode_as_binary_array(raw_message)
        print(f"Binary message length: {len(binary_message)} bits")

        # List to store images and MSE values
        images_with_message = []
        mse_values = []
        nbits_values = list(range(1, 9))  # 1 to 8 bits

        # Create images with different numbers of LSB bits (1-8)
        for nbits in nbits_values:
            # Hide the message
            image_with_message = hide_message(original_image, binary_message, nbits)

            # Save the image
            output_path = f"output_nbits_{nbits}.png"
            save_image(output_path, image_with_message)

            # Calculate MSE
            mse = calculate_mse(original_image, image_with_message)
            mse_values.append(mse)

            # Store the image for display
            images_with_message.append((image_with_message, output_path, nbits))

            # Verify message extraction
            extracted_binary = reveal_message(image_with_message, nbits, len(binary_message))
            extracted_message = decode_from_binary_array(extracted_binary)

            # Verify the first 50 characters match
            print(f"nbits={nbits}, MSE={mse:.2f}")
            print(f"Original start: {raw_message[:50]}")
            print(f"Extracted start: {extracted_message[:50]}")
            print("-" * 50)

        # Create and save the MSE plot
        plt.figure(figsize=(10, 6))
        plt.plot(nbits_values, mse_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of LSB bits used (nbits)')
        plt.ylabel('Mean Square Error (MSE)')
        plt.title('MSE vs. Number of LSB bits')
        plt.grid(True)
        plt.xticks(nbits_values)
        plt.savefig('mse_plot.png')
        print("MSE plot saved to 'mse_plot.png'")

        # Create and save the image comparison
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Comparison of Images with Different LSB Bit Depths', fontsize=16)

        # Flatten the axes for easier indexing
        axes = axes.flatten()

        # Show original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Show modified images
        for i, (img, path, nbits) in enumerate(images_with_message):
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"nbits={nbits}, MSE={mse_values[i]:.2f}")
            axes[i+1].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust for the suptitle
        plt.savefig('all_images_comparison.png')
        print("All images comparison saved to 'all_images_comparison.png'")

        # Display the MSE plot using OpenCV
        mse_plot = cv.imread('mse_plot.png')
        all_images = cv.imread('all_images_comparison.png')

        # Display the plot and images using OpenCV (more reliable than plt.show())
        if mse_plot is not None:
            cv.namedWindow('MSE Plot', cv.WINDOW_NORMAL)
            cv.imshow('MSE Plot', mse_plot)
            cv.resizeWindow('MSE Plot', 800, 600)
            print("Displaying MSE Plot. Press any key to continue...")
            cv.waitKey(0)
            cv.destroyAllWindows()

        if all_images is not None:
            cv.namedWindow('All Images Comparison', cv.WINDOW_NORMAL)
            cv.imshow('All Images Comparison', all_images)
            cv.resizeWindow('All Images Comparison', 1200, 900)
            print("Displaying Image Comparison. Press any key to exit...")
            cv.waitKey(0)
            cv.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
