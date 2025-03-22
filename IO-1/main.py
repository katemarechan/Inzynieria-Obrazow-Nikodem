# main.py

import os
import sys


def clear_screen():
    # Clear screen command based on operating system
    os.system('cls' if os.name == 'nt' else 'clear')


def display_menu():
    print("\n==== Image Processing Tasks ====")
    print("1. High-Pass Filter (Edge Detection)")
    print("2. Color Transformation with Matrix")
    print("3. RGB to YCbCr Conversion")
    print("4. DVB Transmission Simulation")
    print("5. Calculate Mean Square Error")
    print("0. Exit")
    print("Available in the folder images: cat.jpg, catbw.jpg, dexta.jpg, pp.jpg")
    print("===============================")


def main():
    while True:
        clear_screen()
        display_menu()

        choice = input("\nEnter task number (0-5): ")

        if choice == '0':
            print("Exiting program. Goodbye!")
            sys.exit(0)

        elif choice == '1':
            clear_screen()
            print("Running Task 1: High-Pass Filter (Edge Detection)")
            image_path = input("Enter image path (e.g., cat.jpg): ")

            # Import and run Task 1
            import Zad1
            Zad1.high_pass_filter(image_path)

            input("\nPress Enter to return to menu...")

        elif choice == '2':
            clear_screen()
            print("Running Task 2: Color Transformation")
            image_path = input("Enter image path (e.g., cat.jpg): ")

            # Import and run Task 2
            import Zad2
            Zad2.transform_colors(image_path)

            input("\nPress Enter to return to menu...")

        elif choice == '3':
            clear_screen()
            print("Running Task 3: RGB to YCbCr Conversion")
            image_path = input("Enter image path (e.g., cat.jpg): ")

            # Import and run Task 3
            import zad3
            zad3.rgb_to_ycbcr_conversion(image_path)

            input("\nPress Enter to return to menu...")

        elif choice == '4':
            clear_screen()
            print("Running Task 4: DVB Transmission Simulation")
            image_path = input("Enter image path (e.g., cat.jpg): ")

            # Import and run Task 4
            import zad4
            zad4.simulate_dvb_transmission(image_path)

            input("\nPress Enter to return to menu...")

        elif choice == '5':
            clear_screen()
            print("Running Task 5: Calculate Mean Square Error")
            image_path = input("Enter image path (e.g., cat.jpg): ")

            # Import and run Task 5
            import zad5
            zad5.calculate_mse(image_path)

            input("\nPress Enter to return to menu...")

        else:
            print("Invalid choice. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()