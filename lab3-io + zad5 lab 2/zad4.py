#zad4

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


def reveal_message(image, length, nbits=1):

    # Create mask based on number of bits
    mask = (1 << nbits) - 1

    # Flatten the image to process all channels
    if len(image.shape) == 3:  # Color image (RGB)
        flat_image = image.reshape(-1)
    else:  # Grayscale image
        flat_image = image.flatten()

    # Extract bits
    binary_message = ""
    bits_extracted = 0

    for pixel_value in flat_image:
        # Extract the least significant bits
        extracted_bits = pixel_value & mask
        # Convert to binary string
        bit_string = format(extracted_bits, f'0{nbits}b')
        binary_message += bit_string

        bits_extracted += nbits
        if bits_extracted >= length:
            break

    # Return only the required length
    return binary_message[:length]


def extract_hidden_image(image, length, nbits=1):

    # Extract the binary message from the image
    binary_message = reveal_message(image, length, nbits)

    # Convert binary message to hex
    hex_data = ""
    for i in range(0, len(binary_message), 8):
        if i + 8 <= len(binary_message):
            byte = binary_message[i:i + 8]
            hex_data += format(int(byte, 2), '02x')

    # Convert hex to bytes
    bytes_data = bytes.fromhex(hex_data)

    # Write bytes to a file
    output_filename = "extracted_image.jpg"
    with open(output_filename, "wb") as file:
        file.write(bytes_data)

    print(f"Extracted image saved to {output_filename}")

    # Load and return the extracted image for display
    extracted_image = load_image(output_filename)

    return extracted_image


def main():
    """Interactive program to extract a hidden image"""
    print("=== Hidden Image Extraction Program ===")

    try:
        # Get the image file path
        image_path = input("Enter image file with hidden content: ")

        # Get the length of the hidden data
        length = int(input("Enter length of hidden image (in bits): "))

        # Get the number of bits used for encoding
        nbits = int(input("Enter number of least significant bits used for encoding: "))

        # Load the image
        print(f"Loading image from {image_path}...")
        image = load_image(image_path)

        # Extract the hidden image
        print("Extracting hidden image...")
        extracted_image = extract_hidden_image(image, length, nbits)

        # Display the images
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image with hidden content")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(extracted_image)
        plt.title("Extracted hidden image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except ValueError as e:
        print(f"Error: Invalid value. {e}")
    except Exception as e:
        print(f"Error: {e}")

