import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np
from scipy.spatial import Delaunay
import math
import random
import time
from scipy.spatial import ConvexHull


def check_vertice(point: np.array, polygon: np.array) -> bool:
    """
    Checks if a given point is inside the given polygon:
    returns True if inside polygon, returns False if outside polygon

    :param point: np.array([x_coord0, y_coord0])
    :param polygon: np.array([
                        [x_coord0, y_coord0],
                        [x_coord1, y_coord1],
                        [x_coord2, y_coord2],
                        ...,
                        [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
    :return: bool
    """
    if not np.array_equal(polygon[0], polygon[-1]):
        raise ValueError(f"First vertice has to be equal to last vertice: {polygon[0]} != {polygon[-1]}")
    polygon_path = mpath.Path(polygon)
    is_inside = polygon_path.contains_point(point)

    return is_inside


def create_line_vertices(line: np.array, density: float) -> np.array:
    """
    Creates points on a line specified by line. The number of points is given by density which specifies the average
    distance between 2 points (floored!)
    :param line: np.array([x_coord0, y_coord0], [x_coord1, y_coord1])
    :param density: float, specifies distance between 2 points, floor is applied so line = [[0,0],[1,0]], density = 0.4
    returns 3 points instead of 4!
    :return: np.array
    """
    start_point = line[0]
    end_point = line[1]
    distance = np.linalg.norm(start_point - end_point)
    num_subdivisions  = math.floor(distance/density) + 1
    subdivision_points = np.linspace(start_point, end_point, num_subdivisions)

    return subdivision_points


def create_polygon_outline_vertices(polygon: np.array, density: float) -> np.array:
    """
    Creates an array of points with average distance density
    :param polygon: np.array([
                        [x_coord0, y_coord0],
                        [x_coord1, y_coord1],
                        [x_coord2, y_coord2],
                        ...,
                        [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
    :param density: float, specifies distance between 2 points
    :return: np.array
    """
    if not np.array_equal(polygon[0], polygon[-1]):
        raise ValueError(f"First vertice has to be equal to last vertice: {polygon[0]} != {polygon[-1]}")
    outline_vertices = None
    for nv, start_point in enumerate(polygon[:-1]):
        end_point = polygon[nv+1]
        line = np.array([start_point, end_point])
        if nv == 0:
            outline_vertices = create_line_vertices(line, density)[:-1]
        else:
            outline_vertices = np.append(outline_vertices, create_line_vertices(line, density)[:-1], axis=0)

    return outline_vertices


def get_min_max_values(polygon: np.array) -> np.array:
    """
    Returns the min and max x,y values for a polygon (outline positions for rectangle)
    :param polygon: np.array([
                        [x_coord0, y_coord0],
                        [x_coord1, y_coord1],
                        [x_coord2, y_coord2],
                        ...,
                        [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
    :return: np.array([[min_x, min_y], [max_x, max_y]])
    """
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    min_x, max_x = np.min(x_values), np.max(x_values)
    min_y, max_y = np.min(y_values), np.max(y_values)

    return np.array([[min_x, min_y], [max_x, max_y]])


def get_seed_rectangle(rect: np.array, density: float, method='uniform') -> np.array:
    """
    Creates random points in the boundaries of rect  and average distance density
    between points
    :param rect: np.array([[min_x, min_y], [max_x, max_y]])
    :param density: float, specifies distance between 2 points
    :param method: 'random' for random generation, 'uniform' for equal distance or 'randomuniform' for approx uniform
    :return: np.array([[x0, y0], [x1, y1], ... ])
    """
    if np.any(rect < 0.0):
        raise ValueError(f"All vertices have to be positive!")
    if not np.array_equal(rect[0], np.array([0, 0])):
        raise ValueError(f"Starting vortex has to be [0, 0]")

    # randomness, should be close to 1.0
    rd = 0.975
    ru = 1.025

    rect_size_x = np.linalg.norm(rect[1][0] - rect[0][0])
    rect_size_y = np.linalg.norm(rect[1][1] - rect[0][1])
    nbr_points_x = math.floor(rect_size_x/density) + 1
    nbr_points_y = math.floor(rect_size_y/density) + 1

    if method == 'random':
        nbr_points = nbr_points_x * nbr_points_y
        np.random.seed()
        rect_seed_points = np.random.rand(nbr_points, 2)
        rect_seed_points[:, 0] = rect_seed_points[:, 0] * rect_size_x
        rect_seed_points[:, 1] = rect_seed_points[:, 1] * rect_size_y
    elif method == 'uniform':
        x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x)
        y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y)
        x_grid, y_grid = np.meshgrid(x_points, y_points)
        rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    elif method == 'randomuniform':
        x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x-2)
        y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y-2)
        x_grid, y_grid = np.meshgrid(x_points, y_points)
        rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        rect_seed_points = np.array([np.array([point[0] * random.uniform(rd, ru), point[1] * random.uniform(rd, ru)])
                                     for point in rect_seed_points])

    return rect_seed_points


def check_vertice_outline(point: np.array, polygon: np.array, tolerance=1e-6) -> bool:
    """
    Checks if a point is on outline of polyon
    :param point: np.array([x_coord0, y_coord0])
    :param polygon: np.array([
                        [x_coord0, y_coord0],
                        [x_coord1, y_coord1],
                        [x_coord2, y_coord2],
                        ...,
                        [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
    :param tolerance: tolerance for check proximity
    :return: bool
    """
    point_on_line = False
    for nv, start_point in enumerate(polygon[:-1]):
        end_point = polygon[nv+1]
        point_vector = np.array(point) - np.array(start_point)
        line_vector = np.array(end_point) - np.array(start_point)
        cross_product = np.cross(line_vector, point_vector)
        cross_product_length = np.linalg.norm(cross_product)
        is_out_of_line_rect = False
        if (point[0] < start_point[0] or point[0] > end_point[0]) and (point[1] < start_point[1] or point[1] > end_point[0]):
            is_out_of_line_rect = True
        if abs(cross_product_length) < tolerance and not is_out_of_line_rect:
            point_on_line = True
            break

    return point_on_line


def get_seed_polygon(polygon: np.array, density: float, method='uniform'):
    """
    Creates points inside polygon and on polygon outline
    :param polygon: np.array([
                        [x_coord0, y_coord0],
                        [x_coord1, y_coord1],
                        [x_coord2, y_coord2],
                        ...,
                        [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
    :param density: float, specifies distance between 2 points
    :return: np.array
    """
    rect_min_max_coords = get_min_max_values(polygon)
    rect_seed_points = get_seed_rectangle(rect_min_max_coords, density, method=method)

    keep_points = []
    for idn, point in enumerate(rect_seed_points):
        if check_vertice(point, polygon):
            if not check_vertice_outline(point, polygon, tolerance=density/2):
                keep_points.append(idn)
    filtered_seed_points = rect_seed_points[keep_points]
    polygon_outline_vertices = create_polygon_outline_vertices(polygon, density)
    all_points = np.append(filtered_seed_points, polygon_outline_vertices, axis=0)

    return all_points


def show_mesh(all_points, polygon_outline_vertices, triangles):
    """
    todo
    :param all_points:
    :param polygon_outline_vertices:
    :param triangles:
    :return:
    """
    plt.scatter(polygon_outline_vertices[:, 0], polygon_outline_vertices[:, 1], c='b', marker='o', label='Boundary Points')
    plt.scatter(all_points[:, 0], all_points[:, 1], c='b', marker='.', label='Seed Points')
    plt.triplot(all_points[:, 0], all_points[:, 1], triangles, c='gray', label='Mesh')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Mesh generation in Polygon')
    plt.show()


def create_mesh(polygon: np.array, density: float, method='uniform'):
    """
    todo:
    :param polygon:
    :param density:
    :param method:
    :return:
    """
    all_points = get_seed_polygon(polygon, density, method=method)

    # remove duplicates
    all_points = np.unique(all_points, axis=0)

    polygon_outline_vertices = create_polygon_outline_vertices(polygon, density)

    # Triangulation
    triangulation = Delaunay(all_points)
    triangles = triangulation.simplices

    # Remove triangulation outside of polygon
    keep_triangles = []
    for idt, triangle in enumerate(triangles):
        triangle_points = np.array([[all_points[triangle[0]][0], all_points[triangle[0]][1]],
                                    [all_points[triangle[1]][0], all_points[triangle[1]][1]],
                                    [all_points[triangle[2]][0], all_points[triangle[2]][1]]])
        center_point = np.mean(triangle_points, axis=0)
        if check_vertice(center_point, polygon):
            keep_triangles.append(idt)
    triangles_filtered = triangles[keep_triangles]

    return all_points, polygon_outline_vertices, triangles_filtered


def main():
    density = 0.05
    polygon_vertices = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1], [0.5, 0.5], [0, 0.5], [0, 0]])
    polygon_vertices = np.array([[0, 0], [1, 0], [2, 1.2], [0.5, 0.75], [0, 1], [0, 0]])
    start_time = time.time()
    all_points, polygon_outline_vertices, triangles_filtered = create_mesh(polygon_vertices, density, method='randomuniform')
    print(f"Nbr of vertices: {len(all_points)}")
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Function runtime: {runtime} seconds")
    show_mesh(all_points, polygon_outline_vertices, triangles_filtered)


main()