import self_organizing_map

# For plotting the images
from matplotlib import pyplot as plt
import numpy as np

# Training inputs for Building a Grid
training_set = np.array(
    [[0.2, 0.0],
     [0.0, 0.8],
     [1.0, 1.4],
     [1.125, 1.0],
     [1.33, 0.4],
     [1.6, 0.5],
     [0.0, 1.0],
     [1.0, 0.4],
     [1.9, 0.2],
     [0.8, 1.7],
     [0.0, 0.2],
     [0.2, 0.9],
     [0.33,1.33],
     [1.5, 1.5],
     [.66, 1.66]])
#color_names = \
#    ['black', 'blue', 'darkblue', 'skyblue',
#     'greyblue', 'lilac', 'green', 'red',
#     'cyan', 'violet', 'yellow', 'white',
#     'darkgrey', 'mediumgrey', 'lightgrey']

# Train a 20x30 SOM with 400 iterations

som = self_organizing_map.SOM(20, 30, 2, 400)
som.train(training_set)

# Get output grid
image_grid = som.get_centroids()

x = np.array([])
y = np.array([])

for i in range(len(image_grid)):  #seperate x and y values into separate arrays for easy plotting
    x = np.append(x,image_grid[0][i][0])

for i in range(len(image_grid)):
    y = np.append(y, image_grid[0][i][1])

plt.scatter(x ,y)

# Map colours to their closest neurons
# mapped = som.map_vects(training_set)

# Plot
# plt.imshow(image_grid)
# plt.title('Grid SOM')
# for i, m in enumerate(mapped):
#     plt.text(m[1], m[0], color_names[i], ha='center', va='center',
#              bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()