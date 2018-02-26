"""

Basic cairo code to draw a circle

January 28, 2018 

"""

from math import pi
import cairo

#### Accept user inputs

side_length = int(input("Enter the pixel size (128 to 512) of the square canvas on which your image will appear: "))
# while True: #Get canvas dimensions
#     side_length = float(input("Enter the pixel size (128 to 512) of the square canvas on \n which your image will appear: "))
#     if (side_length <= 0)|(side_length > 1)
#         print('The value you entered is not valid (0 < side length < 1)')
#     else
#         break

print("This number will serve to normalize all other measurements")
print("Every other dimension of your objects will be number between 0 and 1")
print("Where 1 represents the length of the entire canvas")

WIDTH, HEIGHT = side_length,side_length

xc = float(input("Enter the x-coordinate for the circle's center: "))

# while True: #Get circle's x location
#     xc = float(input("Enter the x-coordinate for the circle's center: ))
#     if( (xc<=0)|(xc>1)|(not string(xc)))
#         print('The value you entered is not valid (0 < x-center < 1)')
#     else
#         break

yc = float(input("Enter the y-coordinate for the circle's center: "))

# while True: #Get circle's y location
#     yc = float(input("Enter the y-coordinate for the circle's center: ))
#     if( (yc<=0)|(yc>1)|(not string(yc)))
#         print('The value you entered is not valid (0 < y-center < 1)')
#     else
#          break
radius = float(input("Enter the radius of the circle: "))
# while True: #Get circle's radius
#     radius = float(input("Enter the radius of the circle: ))
#     if( (yc<=0)|(yc>1)|(not string(yc)))
#         print('The value you entered is not valid (0 < radius < 1)')
#     else
#         break

#Baseimageinfo for image class

surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context (surface)
ctx.scale (WIDTH, HEIGHT) #Normalizing the canvas
ctx.translate(0.1,0.1) #Changing current transform matrix

#Code that can become a circle class

ctx.arc(xc,yc,radius,0, 2*pi)
ctx.close_path()

ctx.set_source_rgb(0,0,1)
ctx.fill_preserve()
#ctx.set_source_rgb(0,1,0)
#ctx.stroke()

surface.write_to_png ("MyCircle.png")




