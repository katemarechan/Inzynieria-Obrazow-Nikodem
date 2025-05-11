#main.py
import zad1
import zad2
import zad3
import zad4
import zad5
import iminim  # hiding image in image

def show_menu():
    print("[1] LSB Steganography - Basic Message Hiding")
    print("[2] DCT JPEG Steganography")
    print("[3] Multi-Bit LSB Image Steganography")
    print("[4] LSB Steganography with Position Selection")
    print("[5] Extract Hidden Image from Steganography")
    print("[6] Hiding an Image in the Image")
    print("[0] Exit Program")

def main():
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-6): ")

        if choice == "0":
            print("Exiting program.")
            break

        try:
            if choice == "1":
                zad1.main()
            elif choice == "2":
                zad2.main()
            elif choice == "3":
                zad3.main()
            elif choice == "4":
                zad4.main()
            elif choice == "5":
                zad5.main()
            elif choice == "6":
                iminim.main()
            else:
                print("Invalid choice. Please enter a number between 0 and 6.")

        except ImportError as e:
            print(f"Error: Could not import the selected module. Details: {e}")
        except Exception as e:
            print(f"An error occurred while running the selected task. Details: {e}")

if __name__ == "__main__":
    main()
