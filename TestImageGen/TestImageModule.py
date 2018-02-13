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

    def draw_circle(self, radius, x_center, y_center, red, green, blue):
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
