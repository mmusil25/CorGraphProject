# Simple Neural Network for predicting if I patient will
# show up to their appointment based on certain patient
# characteristics. This program was written to develop and display
# my aptitude for machine learning and Python 3.
# Dataset: https://www.kaggle.com/joniarroba/noshowappointments
# Code Guide:
# https://medium.com/rocknnull/playing-with-
# machine-learning-a-practical-example-using-
# keras-tensorflow-790375cd1abb

# Mark Musil B.S.E.E. Portland State University
# March 2018
# ===============================================

import pandas as pds
import datetime

# Pre-processing of the dataset

dataframeX = pds.read_csv('KaggleV2-May-2016.csv', usecols=[
                      2,4, 7, 8, 9, 10, 11, 12, 13])
dataframeY = pds.read_csv('KaggleV2-May-2016.csv', usecols=[5])
print(dataframeX.head())
print(dataframeY.head())

# Convert Gregorian appointment date to a weekday for
# insights on how weekday affects behavior.

def GregorianToWeekday(Gregorian):
    for i in range dataframeX:
        for k in range(len()):




