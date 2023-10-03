import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np

# Define the vertices of the polygon
polygon_vertices = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1], [0.5, 0.5], [0, 0.5], [0, 0]])

# Create a matplotlib path object
polygon_path = mpath.Path(polygon_vertices)

# Define the point to check
point = np.array([0.25, 1])  # Change the values to the coordinates of the point you want to check

# Check if the point is inside the polygon
is_inside = polygon_path.contains_point(point)

# Print the result
if is_inside:
    print(f"The point {point} is inside the polygon.")
else:
    print(f"The point {point} is outside the polygon.")

# Optionally: Plot the polygon and the point for a visual check
plt.plot(polygon_vertices[:, 0], polygon_vertices[:, 1], 'ro-')  # Polygon
plt.plot(point[0], point[1], 'bo')  # Point
plt.show()
