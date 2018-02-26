# Module for the TestImageGen program which contains user input functions as well as
# the OneImage class

from math import pi
import cairo


# The class for one PNG file

class OneImage:

    width = 0  # Image width
    height = 0  # Image height
    label = ""  # Label of image against which the neural network can train

    def __init__(self, width, height, label):
        # Initialize the image with a few mandatory variables.
        self.width = width
        self.height = height
        self.label = label
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.scale(self.width, self.height)

    def draw_circle(self, x_center, y_center, radius, red, green, blue):
        # x_center is the x coordinate of the circle's origin, y_center is the y-coordinate of the circle's origin
        # red, green, blue is the fill color of the circle. The circle has no border color

        self.ctx.arc(x_center, y_center, radius, 0, 2 * pi)
        self.ctx.close_path()  # Complete the circle
        self.ctx.set_source_rgb(red,green,blue)  # Fill color
        self.ctx.fill_preserve()  # Make the circle one color

    def draw_rectangle(self, center_x, center_y, width, height, red, green, blue):

        # This function takes as arguments the top left x and y coordinates of the
        # rectangle which function as the starting position of the rectangle. Width and
        # height are then provided. In addition
        # the fill color of circle is provided via rgb.


        self.ctx.move_to(center_x-(width/2), center_y-(width/2)) # start of the rectangle drawing sequence
        self.ctx.rel_line_to(width, 0)  # right stroke to top right corner using the differential
        # distance function context.rel_line_width(dx,dy)
        self.ctx.rel_line_to(0, height)  # down stroke to bottom right corner
        self.ctx.rel_line_to(-width,0)  # left stroke to bottom left corner
        self.ctx.close_path()  # automatic up stroke to top left corner (beginning of
        # the path
        self.ctx.set_source_rgb(red, green, blue)
        self.ctx.fill_preserve()

    def draw_triangle(self, center_x, center_y, leg, red, green, blue):
        # Draws an equilateral triangle.
        self.ctx.move_to(center_x, center_y - (leg*(3**0.5)))  # Starting Point
        self.ctx.rel_line_to((leg*0.5), leg)
        self.ctx.rel_line_to(-leg, 0)
        self.ctx.close_path()
        self.ctx.set_source_rgb(red, green, blue)
        self.ctx.fill_preserve()

    def write_out_image(self, image_name):
        # image_name must be a string.
        self.surface.write_to_png (image_name)
# =================================================


def manual_circle():  # Allows manual definition of a circle and returns
    # A circle list for use in other areas of the program.

    # Manual Circle parameter input
    circlelist = []
    while True:  # Get circle's x location
        xc = float(input("Enter the x-coordinate for the circle's center:"))
        if xc <= 0 | int(math.floor(xc)) > 512:
            print('The value you entered is not valid (0 < x-center < 1)')
        else:
            circlelist.append(xc)
            break

    while True:  # Get circle's y location
        yc = float(input("Enter the y-coordinate for the circle's center:"))
        if yc <= 0 | int(math.ceil(yc)) > 1:
            print('The value you entered is not valid (0 < y-center < 1)')
        else:
            circlelist.append(yc)
            break

    while True:  # Get circle's radius
        radius = float(input("Enter radius:"))
        if radius <= 0 | int(math.ceil(radius)) > 1:
            print('The value you entered is not valid (0 < radius < 1)')
        else:
            circlelist.append(radius)
            break

    while True:  # Get circle's red color
        red = float(input("Enter the circle's red value (0 <= red value < 1):"))
        if red < 0 | int(math.ceil(red)) > 1:
            print('The value you entered is not valid (0 <= red value < 1)')
        else:
            circlelist.append(red)
            break

    while True:  # Get circle's green color
        green = float(input("Enter the color's green value (0 <= green value < 1):"))
        if green < 0 | int(math.ceil(green)) > 1:
            print('The value you entered is not valid (0 <= green value < 1)')
        else:
            circlelist.append(green)
            break

    while True:  # Get circle's blue color
        blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
        if blue < 0 | int(math.ceil(blue)) > 1:
            print('The value you entered is not valid (0 <= blue value < 1)')
        else:
            circlelist.append(blue)
            break
    mySurface.draw_circle(radius, xc, yc, red, green, blue)
    return circlelist

# ===========================================================


def manual_rectangle():
    # Allow for manual rectangle drawing and returns a list.
    rectanglelist = []
    while True:  # Get x location
        xc = float(input("Enter the x-coordinate for the rectangle's center:"))
        if xc <= 0 | int(math.ceil(xc)) > 1:
            print('The value you entered is not valid (0 < x-center < 1)')
        else:
            rectanglelist.append(xc)
            break

    while True:  # Get y location
        yc = float(input("Enter the y-coordinate for the rectangle's center:"))
        if yc <= 0 | int(math.ceil(yc)) > 1:
            print('The value you entered is not valid (0 < y-center < 1)')
        else:
            rectanglelist.append(yc)
            break

    while True:  # Get rectangle's width
        width = float(input("Enter width:"))
        if width <= 0 | int(math.ceil(width)) > 1:
            print('The value you entered is not valid (0 < width < 1)')
        else:
            rectanglelist.append(width)
            break

    while True:  # Get rectangle's height
        height = float(input("Enter height:"))
        if height <= 0 | int(math.ceil(height)) > 1:
            print('The value you entered is not valid (0 < height < 1)')
        else:
            rectanglelist.append(height)
            break

    while True:  # Get red color
        red = float(input("Enter the red value (0 <= red value < 1):"))
        if red < 0 | int(math.ceil(red)) > 1:
            print('The value you entered is not valid (0 <= red value < 1)')
        else:
            rectanglelist.append(red)
            break

    while True:  # Get green color
        green = float(input("Enter the green value (0 <= green value < 1):"))
        if green < 0 | int(math.ceil(green)) > 1:
            print('The value you entered is not valid (0 <= green value < 1)')
        else:
            rectanglelist.append(green)
            break

    while True:  # Get blue color
        blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
        if blue < 0 | int(math.ceil(blue)) > 1:
            print('The value you entered is not valid (0 <= blue value < 1)')
        else:
            rectanglelist.append(blue)
            break
    mySurface.draw_rectangle(xc, yc, width, height, red, green, blue)
    return rectanglelist

# ===================================================================


def manual_triangle():
    # Triangle input
    trianglelist = []

    while True:  # Get x location
        xc = float(input("Enter the x-coordinate for the triangle's center:"))
        if xc <= 0 | int(math.ceil(xc)) > 1:
            print('The value you entered is not valid (0 < x-center < 1)')
        else:
            trianglelist.append(xc)
            break

    while True:  # Get y location
        yc = float(input("Enter the y-coordinate for the triangle's center:"))
        if yc <= 0 | int(math.ceil(yc)) > 1:
            print('The value you entered is not valid (0 < y-center < 1)')
        else:
            trianglelist.append(yc)
            break

    while True:  # Get leg length
        length = float(input("Enter leg length:"))
        if length <= 0 | int(math.ceil(length)) > 1:
            print('The value you entered is not valid (0 < leg length < 1)')
        else:
            trianglelist.append(length)
            break

    while True:  # Get red color
        red = float(input("Enter the red value (0 <= red value < 1):"))
        if red < 0 | int(math.ceil(red)) > 1:
            print('The value you entered is not valid (0 <= red value < 1)')
        else:
            trianglelist.append(red)
            break

    while True:  # Get green color
        green = float(input("Enter the green value (0 <= green value < 1):"))
        if green < 0 | int(math.ceil(green)) > 1:
            print('The value you entered is not valid (0 <= green value < 1)')
        else:
            trianglelist.append(green)
            break

    while True:  # Get blue color
        blue = float(input("Enter the color's blue value (0 <= blue value < 1):"))
        if blue < 0 | int(math.ceil(blue)) > 1:
            print('The value you entered is not valid (0 <= blue value < 1)')
        else:
            trianglelist.append(blue)
            break

    mySurface.draw_triangle(xc, yc, length, red, green, blue)
    return trianglelist

