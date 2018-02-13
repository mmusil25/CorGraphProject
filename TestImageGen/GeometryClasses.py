# Module for the TestImageGen program which contains user input functions as well as
# the OneImage class

import TestImageModule
import math
# -------------------- #
# Accept user inputs   #
# -------------------- #

# What is the size of your image? (128,256, 512)
# What geometric shapes would you like on your image?
# Define the center point, radius, and r,g,b value for the circle.
# Define the center point, length, with, and r,g,b values for the rectangle.
# Define the center point, leg length, r,g,b values for the triangle.
# Define the file location for this to be written out to.
# Define the label for your image (also serves as the final name at this point).


while True: #Get canvas dimensions
    SideLength = int(input("Enter the pixel size (128 to 512) of the square"
                           " canvas on which your image will appear: "))
    if (SideLength <= 128)|(SideLength > 512):
         print('The value you entered is not valid (0 < side length < 1)')
    else:
        break

print("This number will serve to normalize all other measurements")
print("Every other dimension of your image including shape dimensions"
      " will be number between 0 and 1 as a factor of side length")
print("Where 1 represents the length of the entire canvas")

#print("Which objects would you like on your image?")
#print("(1) Triangle, (2) Rectangle, (3) Circle, (4) T & R, (5) T & C, (6) R & C, (7) T & R & C")


mySurface = TestImageModule.OneImage(SideLength, SideLength, "TestLabel")

# Circle input

while True:  # Get circle's x location
    xc = float(input("Enter the x-coordinate for the circle's center:"))
    if xc <= 0 | int(math.floor(xc)) > 512:
        print('The value you entered is not valid (0 < x-center < 1)')
    else:
        break

while True:  # Get circle's y location
    yc = float(input("Enter the y-coordinate for the circle's center:"))
    if yc <= 0 | int(math.ceil(yc)) > 1:
        print('The value you entered is not valid (0 < y-center < 1)')
    else:
        break

while True:  # Get circle's radius
    radius = float(input("Enter radius:"))
    if radius <= 0 | int(math.ceil(radius)) > 1:
        print('The value you entered is not valid (0 < radius < 1)')
    else:
        break

while True:  # Get circle's red color
    red = float(input("Enter the circle's red value (0 <= red value < 1):"))
    if red < 0 | int(math.ceil(red)) > 1:
        print('The value you entered is not valid (0 <= red value < 1)')
    else:
        break

while True:  # Get circle's green color
    green = float(input("Enter the color's green value (0 <= green value < 1):"))
    if green < 0 | int(math.ceil(green)) > 1:
        print('The value you entered is not valid (0 <= green value < 1)')
    else:
        break

while True:  # Get circle's blue color
    blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
    if blue < 0 | int(math.ceil(blue)) > 1:
        print('The value you entered is not valid (0 <= blue value < 1)')
    else:
        break
mySurface.draw_circle(radius, xc, yc, red, green, blue)

# Rectangle input

while True:  # Get x location
    xc = float(input("Enter the x-coordinate for the rectangle's center:"))
    if xc <= 0 | int(math.ceil(xc)) > 1:
        print('The value you entered is not valid (0 < x-center < 1)')
    else:
        break

while True:  # Get y location
    yc = float(input("Enter the y-coordinate for the rectangle's center:"))
    if yc <= 0 | int(math.ceil(yc)) > 1:
        print('The value you entered is not valid (0 < y-center < 1)')
    else:
        break

while True:  # Get rectangle's width
    width = float(input("Enter width:"))
    if width <= 0 | int(math.ceil(width)) > 1:
        print('The value you entered is not valid (0 < width < 1)')
    else:
        break

while True:  # Get rectangle's height
    height = float(input("Enter height:"))
    if height <= 0 | int(math.ceil(height)) > 1:
        print('The value you entered is not valid (0 < height < 1)')
    else:
        break

while True:  # Get red color
    red = float(input("Enter the red value (0 <= red value < 1):"))
    if red < 0 | int(math.ceil(red)) > 1:
        print('The value you entered is not valid (0 <= red value < 1)')
    else:
        break

while True:  # Get green color
    green = float(input("Enter the green value (0 <= green value < 1):"))
    if green < 0 | int(math.ceil(green)) > 1:
        print('The value you entered is not valid (0 <= green value < 1)')
    else:
        break

while True:  # Get blue color
    blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
    if blue < 0 | int(math.ceil(blue)) > 1:
        print('The value you entered is not valid (0 <= blue value < 1)')
    else:
        break
mySurface.draw_rectangle(xc, yc, width, height, red, green, blue)

# Triangle input

while True:  # Get x location
    xc = float(input("Enter the x-coordinate for the triangle's center:"))
    if xc <= 0 | int(math.ceil(xc)) > 1:
        print('The value you entered is not valid (0 < x-center < 1)')
    else:
        break

while True:  # Get y location
    yc = float(input("Enter the y-coordinate for the triangle's center:"))
    if yc <= 0 | int(math.ceil(yc)) > 1:
        print('The value you entered is not valid (0 < y-center < 1)')
    else:
        break

while True:  # Get leg length
    length = float(input("Enter leg length:"))
    if length <= 0 | int(math.ceil(length)) > 1:
        print('The value you entered is not valid (0 < leg length < 1)')
    else:
        break

while True:  # Get red color
    red = float(input("Enter the red value (0 <= red value < 1):"))
    if red < 0 | int(math.ceil(red)) > 1:
        print('The value you entered is not valid (0 <= red value < 1)')
    else:
        break

while True:  # Get green color
    green = float(input("Enter the green value (0 <= green value < 1):"))
    if green < 0 | int(math.ceil(green)) > 1:
        print('The value you entered is not valid (0 <= green value < 1)')
    else:
        break

while True:  # Get blue color
    blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
    if blue < 0 | int(math.ceil(blue)) > 1:
        print('The value you entered is not valid (0 <= blue value < 1)')
    else:
        break

mySurface.draw_triangle(xc, yc, length, red, green, blue)

while True:  # Get image label
    label = input("Enter the image's label") + ".PNG"
    break

mySurface.write_out_image(label)