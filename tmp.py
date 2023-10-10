import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri

# Generate some random data
np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
z = x + y

# Create a Delaunay triangulation
triang = tri.Triangulation(x, y)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the contour filled with the z values
contour = ax.tricontourf(triang, z, cmap='viridis')

# Plot the scatter points
scatter = ax.scatter(x, y, c=z, cmap='viridis', marker='o', edgecolors='k')

# Plot the triangle edges using triplot
triplot = ax.triplot(triang, 'r-', linewidth=0.5)

# Add color bar for the scatter plot
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Z Values')

# Add labels and a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contourf, Scatter, and Triplot Example')

# Show the plot
plt.show()
