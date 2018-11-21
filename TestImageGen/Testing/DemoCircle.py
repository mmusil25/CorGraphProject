# -*- coding: utf-8 -*-
"""
circle_contexteated on Tue Jan 23 16:22:22 2018

Program which uses the command Line to draw a circle

@author: mark
"""
#####Set up  ######

#import modules
from math import pi
import cairo

#circle_contexteate cairo surface and context objects
surface = cairo.SVGSurface("circle.svg", 200, 200)
circle_context = cairo.Context(surface) # An object of the context class instantiated using 
#the object surface as an initialization parameter.

#Define a few drawing functions

def path_ellipse(circle_context, x, y, width, height, angle=0):
    """
    x      - center x
    y      - center y
    width  - width of ellipse  (in x direction when angle=0)
    height - height of ellipse (in y direction when angle=0)
    angle  - angle in radians to rotate, clockwise
    """
    circle_context.save()
    circle_context.translate(x, y)
    circle_context.rotate(angle)
    circle_context.scale(width / 2.0, height / 2.0)
    circle_context.arc(0.0, 0.0, 1.0, 0.0, 2.0 * pi)
    circle_context.restore()

def draw(circle_context, width, height):
    circle_context.scale(width, height)
    circle_context.set_line_width(0.04)

    path_ellipse(circle_context, 0.5, 0.5, 1.0, 0.3, pi / 4.0)

    # fill
    circle_context.set_source_rgba(1, 0, 0, 1)
    circle_context.fill_preserve()

    # stroke
    # reset identity matrix so line_width is a constant
    # width in device-space, not user-space
    circle_context.save()
    circle_context.identity_matrix()
    circle_context.set_source_rgba(0, 0, 0, 1)
    circle_context.set_line_width(3)
    circle_context.stroke()
    circle_context.restore()
    surface.finish()
    


###Interactive Portion###
x = float(raw_input("Define the x-value of the center of the ellipse: "))
y = float(raw_input("Define the y-value of the center of the ellipse: "))
width = float(raw_input("Define the width of the ellipse "))
height = float(raw_input("Define the hieght of the ellipse"))
path_ellipse(circle_context,x,y,width,height)
draw(circle_context,width,height)
