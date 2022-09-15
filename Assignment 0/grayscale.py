#####################################################################
# Omar Abdelmotaleb
# I pledge my honor that I have abided by the Stevens Honor System.
# Part 2: grayscale.py
#####################################################################

from matplotlib.image import imread
from matplotlib import pyplot as plt

# Takes in string input of a file. Ex: input.jpg

infile = str(input())

image_rgb = imread(infile)

# Iterate through to get every pixel (x,y)
for x in range(len(image_rgb)):
    for y in range(len(image_rgb[x])):
        r = image_rgb[x][y][0]
        g = image_rgb[x][y][1]
        b = image_rgb[x][y][2]
        # Apply given equation to determine grayscale value
        value = 0.2989*r + 0.5870*g + 0.1140*b
        image_rgb[x][y] = value

# Save it as outfile.jpg
image_gray = plt.imsave('outfile.jpg', image_rgb/255)