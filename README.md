  Please remember to create your exe file.
  
  I personally had troubles with it, remember to add all the libraries to your program, hidden imports, include the pictures into the dist folder. If youre doing it in PyCharm and its not working through pip - try   the broader approach of doing it as, for example
  
  python -m PyInstaller --onefile --name SteganographyToolkit --hidden-import numpy --hidden-import cv2 --hidden-import matplotlib --hidden-import matplotlib.pyplot --hidden-import scipy --hidden-import scipy.fftpack --hidden-import PIL --hidden-import binascii --hidden-import math --add-data "zad1.py;." --add-data "zad2.py;." --add-data "zad3.py;." --add-data "zad4.py;." --add-data "zad5.py;." --add-data "iminim.py;." main.py (for lab 4)
  
  The -m Pyinstaller may be important for fixing your problems if the program cant find some files. I also recommend Python 11 configurator instead of Python 10 since it works smoother and works better with some of the needed libraries.
  
  Also remember to not write "if name == __main__ " etc on the end of your files if you want to create a menu file later, i also ran into a little problem after forgetting to delete them after testing the individual files and tasks. Stupid mistake, please dont repeat them, i was tired, learn from my mistakes, dont repeat them.
  
  If you have any problems or questions -- feel free to ask, im always ready to help and answer. 
