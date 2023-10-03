import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

# Define the size of the square
square_size = 1.0

# Generate random points inside the square
np.random.seed()
points = np.random.rand(100, 2) * square_size
print(points)
# Perform Delaunay triangulation
triangulation = Delaunay(points)

# Plot the original points
plt.scatter(points[:, 0], points[:, 1], c='b', marker='o', label='Points')

# Plot the square boundary
plt.plot([0, square_size, square_size, 0, 0], [0, 0, square_size, square_size, 0], 'k--', label='Square Boundary')

# Plot the triangles
plt.triplot(points[:, 0], points[:, 1], triangulation.simplices, c='r', label='Delaunay Triangulation')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Delaunay Triangulation Example Inside a Square')
plt.xlim(0, square_size)
plt.ylim(0, square_size)
plt.show()
