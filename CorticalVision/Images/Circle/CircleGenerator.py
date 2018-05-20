# Module for the TestImageGen program which contains user input functions as well as
# the OneImage class

import TestImageGenModule
import math
import numpy
import cairo
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


# while True:  # Get canvas dimensions
#     SideLength = int(input("Enter the pixel size (16 to 512) of the square"
#                            " canvas on which your image will appear: "))
#     if (SideLength <= 16)|(SideLength > 512):
#         print('The value you entered is not valid (0 < side length < 1)')
#     else:
#         break
#
# print("This number will serve to normalize all other measurements")
# print("Every other dimension of your image including shape dimensions"
#       " will be number between 0 and 1 as a factor of side length")
# print("Where 1 represents the length of the entire canvas")
#
# while True:  # Get Number of images to be generated
#     ImageQuantity = int(input("Enter the number of images (1 to 512) to be generated: "))
#     if (ImageQuantity <= 1)|(ImageQuantity > 512):
#         print('The value you entered is not valid (0 < Image Quantity <= 512)')
#     else:
#         break
#
# while True:  # Get deltaA area variance interval for each geometric shape
#     deltaA = float(input("Enter the area variance interval(0.0 to 1) to be used: "))
#     if (math.floor(deltaA) < 0)|(math.ceil(deltaA) > 1):
#         print('The value you entered is not valid (0 < deltaA <= 1)')
#     else:
#         break
# print("Which objects would you like on your image?")
# print("(1) Triangle, (2) Rectangle, (3) Circle, (4) T & R, (5) T & C, (6) R & C, (7) T & R & C")

# mySurface.manual_circle()
# mySurface.manual_rectangle()
# mySurface.manual_triangle()

# while True:  # Get image label
#     label = input("Enter the image's label") + ".PNG"
#     break

SideLength, ImageQuantity, deltaA = 312, 500, 0.35
deltaX, deltaY, circleScale = 0.13, 0.1, 10

for i in range(ImageQuantity):
    radius = 0.19  # Geometric shape area seeds
    width = 0.2
    height = width
    leg = 0.15
    label = "C" + str(i) + ".PNG"
    # Generate random rgb values for the background
    low,high = 0.35,0.9
    backGroundR = numpy.random.uniform(0.0,low)
    backGroundG = numpy.random.uniform(0.0,low)
    backGroundB = numpy.random.uniform(0.0,low)
    # Generate rangom rgb for the shapes
    objectR = numpy.random.uniform(high,1)
    objectG = numpy.random.uniform(high,1)
    objectB = numpy.random.uniform(high,1)

    mySurface = TestImageGenModule.OneImage(SideLength, SideLength, label)
    mySurface.ctx.set_source_rgb(backGroundR, backGroundG, backGroundB)
    mySurface.ctx.paint()
    # ==== Circle generation ====
    xc = 0.5  # Object center seeds
    yc = xc
    xc += numpy.random.uniform(-deltaX, deltaX)
    yc += numpy.random.uniform(-deltaY, deltaY)
    radius += numpy.random.uniform(-deltaA/circleScale, deltaA/circleScale)
    mySurface.draw_circle(xc, yc, radius, objectR, objectG, objectB)
    # ==== Rectangle generation ====
    xc = 0.5  # Object center seeds
    yc = xc
    xc += numpy.random.uniform(-deltaX, deltaX)
    yc += numpy.random.uniform(-deltaY, deltaY)
    width += numpy.random.uniform(-deltaA/8, deltaA)
    height = width
    # height += numpy.random.uniform(-deltaA, deltaA)
    # mySurface.draw_rectangle(xc, yc, width, height, objectR, objectG, objectB)
    # ==== Triangle generation ====
    xc = 0.5 # Object center seeds
    yc = xc
    xc += numpy.random.uniform(-deltaX, deltaX)
    yc += numpy.random.uniform(-deltaY, deltaY)
    leg += numpy.random.uniform(-deltaA/8, deltaA)
    #mySurface.draw_triangle(xc, yc, leg, objectR, objectG, objectB)
    mySurface.write_out_image(label)



