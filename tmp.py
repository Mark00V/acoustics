import numpy as np


def is_point_on_line(point, line_start, line_end, tolerance=1e-6):
    # Calculate vectors from line_start to point and from line_start to line_end
    point_vector = np.array(point) - np.array(line_start)
    line_vector = np.array(line_end) - np.array(line_start)

    # Calculate the cross product of the two vectors
    cross_product = np.cross(line_vector, point_vector)

    # Calculate the length of the cross product
    cross_product_length = np.linalg.norm(cross_product)

    # Check if the length of the cross product is within the specified tolerance
    return abs(cross_product_length) < tolerance


# Define the points that make up the line
line_start = (1, 1)
line_end = (4, 4)

# Specify the point you want to check
point_to_check = (2, 2)

# Check if the point is on the line
is_on_line = is_point_on_line(point_to_check, line_start, line_end)

if is_on_line:
    print("The point is on the line.")
else:
    print("The point is not on the line.")
