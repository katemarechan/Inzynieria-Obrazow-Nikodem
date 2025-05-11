#zad5

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


def hide_message(image, message, nbits=1):

    # Make a copy of the image to avoid modifying the original
    stego_image = image.copy()

    # Create a bit mask for clearing the LSBs
    mask = ~((1 << nbits) - 1)

    # Flatten the image to process all channels
    if len(stego_image.shape) == 3:  # Color image (RGB)
        flat_image = stego_image.reshape(-1)
    else:  # Grayscale image
        flat_image = stego_image.flatten()

    # Check if the image is large enough to hide the message
    if len(message) > flat_image.size * nbits:
        raise ValueError(f"Message too large to hide in this image. Maximum length: {flat_image.size * nbits} bits")

    # Embed message bits
    for i in range(0, len(message), nbits):
        if i + nbits <= len(message):
            pixel_index = i // nbits
            if pixel_index < len(flat_image):
                # Clear the LSBs
                flat_image[pixel_index] = flat_image[pixel_index] & mask

                # Get nbits from the message
                message_bits = message[i:i + nbits]
                message_int = int(message_bits, 2)

                # Set the LSBs
                flat_image[pixel_index] = flat_image[pixel_index] | message_int

    return stego_image


def reveal_message_until_footer(image, nbits=1):

    # Common JPEG file signature (Start Of Image marker)
    jpeg_header_hex = "FFD8FF"
    jpeg_header_bin = ''.join(
        [format(int(jpeg_header_hex[i:i + 2], 16), '08b') for i in range(0, len(jpeg_header_hex), 2)])

    # Common JPEG file footer (End Of Image marker)
    jpeg_footer_hex = "FFD9"
    jpeg_footer_bin = ''.join(
        [format(int(jpeg_footer_hex[i:i + 2], 16), '08b') for i in range(0, len(jpeg_footer_hex), 2)])

    # Create mask based on number of bits
    mask = (1 << nbits) - 1

    # Flatten the image to process all channels
    if len(image.shape) == 3:  # Color image (RGB)
        flat_image = image.reshape(-1)
    else:  # Grayscale image
        flat_image = image.flatten()

    # Extract bits
    binary_message = ""

    # Check if we have the header and footer in our extracted data
    header_found = False
    footer_pos = -1

    # Extract all data from the image
    for pixel_value in flat_image:
        # Extract the least significant bits
        extracted_bits = pixel_value & mask
        # Convert to binary string
        bit_string = format(extracted_bits, f'0{nbits}b')
        binary_message += bit_string

        # After extracting enough bits to potentially contain a header
        if len(binary_message) >= len(jpeg_header_bin) and not header_found:
            if binary_message[-len(jpeg_header_bin):] == jpeg_header_bin:
                header_found = True
                # Keep only the header part and what follows
                binary_message = binary_message[-len(jpeg_header_bin):]
                print("JPEG header found!")

        # Check for footer once header is found
        if header_found and len(binary_message) >= len(jpeg_header_bin) + len(jpeg_footer_bin):
            # Look for footer in the most recent portion of the message
            search_start = max(0, len(binary_message) - len(jpeg_footer_bin) - 32)  # Search in the last part
            search_portion = binary_message[search_start:]

            footer_index = search_portion.find(jpeg_footer_bin)
            if footer_index != -1:
                # Calculate the actual position in the full message
                footer_pos = search_start + footer_index + len(jpeg_footer_bin)
                print(f"JPEG footer found! Extracted {footer_pos} bits.")
                break

    # If we found a valid header and footer
    if header_found and footer_pos > 0:
        return binary_message[:footer_pos]

    # If we didn't find a clear header/footer but extracted data
    if len(binary_message) > 0:
        # Look for the JPEG footer in the entire binary message
        footer_index = binary_message.find(jpeg_footer_bin)
        if footer_index != -1:
            return binary_message[:footer_index + len(jpeg_footer_bin)]

        # Look for JPEG header in the entire binary message
        header_index = binary_message.find(jpeg_header_bin)
        if header_index != -1:
            # Take everything from the header to the end,
            # assuming it's a truncated image
            return binary_message[header_index:]

    # If all else fails, return everything and let the image decoder handle it
    print("Warning: Could not detect JPEG header/footer. Returning all extracted data.")
    return binary_message


def extract_hidden_image(image, nbits=1):

    # Extract the binary message from the image
    binary_message = reveal_message_until_footer(image, nbits)

    print(f"Extracted {len(binary_message)} bits of data")

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

    print(f"Extracted image saved to {output_filename} ({len(bytes_data)} bytes)")

    try:
        # Load and return the extracted image for display
        extracted_image = load_image(output_filename)
        return extracted_image
    except Exception as e:
        print(f"Warning: Could not load the extracted image: {e}")
        print("The data may not form a valid image.")
        return None


def hide_image(image, secret_image_path, nbits=1):

    with open(secret_image_path, "rb") as file:
        secret_img = file.read()

    # Convert bytes to hex string
    secret_img = secret_img.hex()

    # Split hex string into 2-character chunks
    secret_img = [secret_img[i:i + 2] for i in range(0, len(secret_img), 2)]

    # Convert each hex chunk to 8-bit binary
    secret_img = ["{:08b}".format(int(el, base=16)) for el in secret_img]

    # Join all binary strings
    secret_img = "".join(secret_img)

    # Hide the binary message in the image
    return hide_message(image, secret_img, nbits), len(secret_img)


def main():
    """Interactive program to extract a hidden image"""
    print("=== Hidden Image Extraction Program ===")

    try:
        # Get the image file path
        image_path = input("Enter image file with hidden content: ")

        # Get the number of bits used for encoding
        nbits = int(input("Enter number of least significant bits used for encoding: "))

        # Load the image
        print(f"Loading image from {image_path}...")
        image = load_image(image_path)

        # Extract the hidden image
        print("Extracting hidden image...")
        extracted_image = extract_hidden_image(image, nbits)

        if extracted_image is not None:
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
        else:
            print("Could not display the extracted image.")

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except ValueError as e:
        print(f"Error: Invalid value. {e}")
    except Exception as e:
        print(f"Error: {e}")

