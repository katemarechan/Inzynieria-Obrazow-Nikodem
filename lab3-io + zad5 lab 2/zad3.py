#zad3
from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
import os

plt.rcParams["figure.figsize"] = (18,10)

def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = ["{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)

def decode_from_binary_array(array):
    """Decode a binary string to utf8."""
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = ["{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")

def load_image(path, pad=False):
    """Load an image.
    If pad is set then pad an image to multiple of 8 pixels.
    """
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
            image, ((0, y_pad), (0, x_pad), (0, 0)), mode='constant')
    return image

def save_image(path, image):
    """Save an image."""
    plt.imsave(path, image)

def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)

def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB) starting from a specific position.
    nbits: number of least significant bits
    spos: starting position in the flattened image array
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    # Make sure starting position is within the image bounds
    spos = clamp(spos, 0, len(image) - 1)

    # Check if message will fit in the image from the starting position
    if len(message) > (len(image) - spos) * nbits:
        raise ValueError("Message is too long for the specified starting position :(")

    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
    for i, chunk in enumerate(chunks):
        # Add spos to position to start from the specified position
        pos = spos + i

        # Make sure we're still within the image bounds
        if pos >= len(image):
            break

        # Pad chunk if it's shorter than nbits
        if len(chunk) < nbits:
            chunk = chunk.ljust(nbits, '0')

        byte = "{:08b}".format(image[pos])
        new_byte = byte[:-nbits] + chunk
        image[pos] = int(new_byte, 2)

    return image.reshape(shape)

def reveal_message(image, nbits=1, length=0, spos=0):
    """Reveal the hidden message starting from a specific position.
    nbits: number of least significant bits
    length: length of the message in bits
    spos: starting position in the flattened image array
    """
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()

    # Make sure starting position is within the image bounds
    spos = clamp(spos, 0, len(image) - 1)

    # Calculate how many pixels we need to check
    if length <= 0:
        length_in_pixels = len(image) - spos
    else:
        length_in_pixels = math.ceil(length/nbits)

    # Make sure we don't go beyond the image bounds
    if spos + length_in_pixels > len(image):
        length_in_pixels = len(image) - spos

    message = ""
    i = 0
    while i < length_in_pixels:
        # Add spos to position to start from the specified position
        pos = spos + i

        # Make sure we're still within the image bounds
        if pos >= len(image):
            break

        byte = "{:08b}".format(image[pos])
        message += byte[-nbits:]
        i += 1

    # If a specific length was requested, trim to that length
    if length > 0:
        mod = length % nbits
        if mod != 0:
            message = message[:-nbits+mod]
        else:
            message = message[:length]

    return message

# Ask user for the image path
def get_image_path():
    while True:
        path = input("Enter the path to the image file: ")
        if os.path.exists(path):
            return path
        else:
            print(f"Error: File '{path}' does not exist. Please try again.")

def main():
    """Main function to demonstrate steganography capabilities."""
    # Get image path from user
    image_path = get_image_path()

    # Set the message
    message = input("Enter the message to hide: ")
    if not message:
        message = "LA PASSION"
        print(f"Using default message: {message}")

    # Load the original image
    original_image = load_image(image_path)
    print(f"Original image dimensions: {original_image.shape}")

    # Convert message to binary
    binary_message = encode_as_binary_array(message)
    print(f"Message: '{message}'")
    print(f"Binary message: {binary_message}")
    print(f"Binary message length: {len(binary_message)} bits")

    # Define starting position (e.g., 1000 pixels from the beginning)
    start_position = 1000
    print(f"Starting position for hiding message: {start_position}")

    # Hide message in the image (using 1 bit) starting from position 1000
    nbits = 1
    image_with_message = hide_message(original_image, binary_message, nbits, start_position)

    # Save the image with hidden message
    output_path = "output.png"
    save_image(output_path, image_with_message)
    print(f"Image with hidden message saved to '{output_path}'")

    # Load the saved image
    loaded_image = load_image(output_path)

    # Reveal the hidden message from the same starting position
    extracted_binary = reveal_message(loaded_image, nbits, len(binary_message), start_position)
    extracted_message = decode_from_binary_array(extracted_binary)

    # Print results
    print(f"Extracted message: {extracted_message}")

    # Display images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_with_message)
    plt.title("Image with Hidden Message")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
