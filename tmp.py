import numpy as np

# Define the coordinates of the rectangle
x_min, x_max = 1.0, 5.0  # Minimum and maximum x-coordinates
y_min, y_max = 2.0, 6.0  # Minimum and maximum y-coordinates

# Specify the number of points along the x and y axes
num_x_points = 10  # Number of points along the x-axis
num_y_points = 5   # Number of points along the y-axis

# Generate a grid of points
x_points = np.linspace(x_min, x_max, num_x_points)
y_points = np.linspace(y_min, y_max, num_y_points)

# Create a meshgrid of x and y points
x_grid, y_grid = np.meshgrid(x_points, y_points)

# Stack the x and y coordinates to get the final array of points
points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

# Print the generated points
print("Generated points:")
print(points)
