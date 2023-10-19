import numpy as np
import matplotlib.path as mpath
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


class CreateMesh:
    """
    Creates an array of mesh points and triangulation
    """

    def __init__(self, polygon_vertices: np.array, density: float, method='uniform'):
        """
        Initializes a Polygon object with the given polygon vertices and density.

        Args:
        :param density: float, specifies distance between 2 points, floor is applied so line = [[0,0],[1,0]], density = 0.4
        returns 3 points instead of 4!
        :param polygon: np.array([
                            [x_coord0, y_coord0],
                            [x_coord1, y_coord1],
                            [x_coord2, y_coord2],
                            ...,
                            [x_coord0, y_coord0]]) # Last vertice has to be first vertice!
        :param method: optional, 'random' for random generation, 'uniform' for equal distance or 'randomuniform' for approx uniform

        Raises:
            ValueError
        Returns:
            None
        """
        self.polygon = polygon_vertices
        self.density = density
        self.method = method
        self.all_points = None
        self.polygon_outline_vertices = None
        self.triangles = None
        self.meshcreated = False
        if not np.array_equal(self.polygon[0], self.polygon[-1]):
            raise ValueError(f"First vertice has to be equal to last vertice: {self.polygon[0]} != {self.polygon[-1]}")

    def check_vertice(self, point: np.array) -> bool:
        """
        Checks if a given point is inside self.polygon:
        returns True if inside polygon, returns False if outside polygon

        Args:
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """

        polygon_path = mpath.Path(self.polygon)
        is_inside = polygon_path.contains_point(point)

        return is_inside

    def check_vertice_polygon(self, point: np.array, polygon: np.array) -> bool:
        """
        Checks if a given point is inside the given polygon:
        returns True if inside polygon, returns False if outside polygon

        Args:
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """

        polygon_path = mpath.Path(polygon)
        is_inside = polygon_path.contains_point(point)

        return is_inside

    def create_line_vertices(self, line: np.array) -> np.array:
        """
        Creates points on a line specified by line. The number of points is given by density which specifies the average
        distance between 2 points (floored!)
        :param line: np.array([x_coord0, y_coord0], [x_coord1, y_coord1])
        :return: np.array
        """

        start_point = line[0]
        end_point = line[1]
        distance = np.linalg.norm(start_point - end_point)
        num_subdivisions = math.floor(distance / self.density) + 1
        subdivision_points = np.linspace(start_point, end_point, num_subdivisions)

        return subdivision_points

    def create_polygon_outline_vertices(self) -> np.array:
        """
        Creates an array of points with average distance density
        Args:
        Nothing
        :return: np.array
        """

        if not np.array_equal(self.polygon[0], self.polygon[-1]):
            raise ValueError(f"First vertice has to be equal to last vertice: {self.polygon[0]} != {self.polygon[-1]}")

        outline_vertices = None

        boundaries = list()
        n_point = 0
        corner_nodes = list()
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            line = np.array([start_point, end_point])
            if nv == 0:
                outline_vertices = self.create_line_vertices(line)[:-1]
                corner_nodes.append([n_point, outline_vertices[0]])
                boundary_points = list()
                for vertice in outline_vertices:
                    boundary_points.append([n_point, vertice])
                    n_point += 1
                boundaries.append(boundary_points)
            else:
                this_boundary = self.create_line_vertices(line)[:-1]
                corner_nodes.append([n_point, this_boundary[0]])
                boundary_points = list()
                for vertice in this_boundary:
                    boundary_points.append([n_point, vertice])
                    n_point += 1
                boundaries.append(boundary_points)
                outline_vertices = np.append(outline_vertices, this_boundary, axis=0)

        boundaries_numbered = boundaries
        all_outline_vertices_numbered = [[n_point, outline_vertices[n_point]] for n_point in
                                         range(len(outline_vertices))]

        print(corner_nodes)

        return all_outline_vertices_numbered, boundaries_numbered

    def get_min_max_values(self) -> np.array:
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

        x_values = self.polygon[:, 0]
        y_values = self.polygon[:, 1]
        min_x, max_x = np.min(x_values), np.max(x_values)
        min_y, max_y = np.min(y_values), np.max(y_values)

        return np.array([[min_x, min_y], [max_x, max_y]])

    def get_seed_rectangle(self, rect: np.array) -> np.array:
        """
        Creates random points in the boundaries of rect  and average distance density
        between points
        :param rect: np.array([[min_x, min_y], [max_x, max_y]])
        :return: np.array([[x0, y0], [x1, y1], ... ])
        """

        if np.any(rect < 0.0):
            raise ValueError(f"All vertices have to be positive!")
        if not np.array_equal(rect[0], np.array([0, 0])):
            raise ValueError(f"Starting vortex has to be [0, 0]")

        # randomness, should be close to 1.0
        rd = 0.99
        ru = 1.01

        rect_size_x = np.linalg.norm(rect[1][0] - rect[0][0])
        rect_size_y = np.linalg.norm(rect[1][1] - rect[0][1])
        nbr_points_x = math.floor(rect_size_x / self.density) + 1
        nbr_points_y = math.floor(rect_size_y / self.density) + 1

        if self.method == 'random':
            nbr_points = nbr_points_x * nbr_points_y
            np.random.seed()
            rect_seed_points = np.random.rand(nbr_points, 2)
            rect_seed_points[:, 0] = rect_seed_points[:, 0] * rect_size_x
            rect_seed_points[:, 1] = rect_seed_points[:, 1] * rect_size_y
        elif self.method == 'uniform':
            x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x)
            y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y)
            x_grid, y_grid = np.meshgrid(x_points, y_points)
            rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        elif self.method == 'randomuniform':
            x_points = np.linspace(rect[0][0], rect[1][0], nbr_points_x - 2)
            y_points = np.linspace(rect[0][1], rect[1][1], nbr_points_y - 2)
            x_grid, y_grid = np.meshgrid(x_points, y_points)
            rect_seed_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
            rect_seed_points = np.array(
                [np.array([point[0] * random.uniform(rd, ru), point[1] * random.uniform(rd, ru)])
                 for point in rect_seed_points])

        return rect_seed_points

    def draw_quadrilateral(self, corners: np.array):
        """
        For testing purposes
        :return:
        """

        plt.scatter(corners[:, 0], corners[:, 1], c='b', marker='x', label='Corners')
        plt.show()

    def check_vertice_outline(self, point: np.array) -> bool:
        """
        Checks if a point is on outline of polyon
        :param point: np.array([x_coord0, y_coord0])
        :return: bool
        """

        tolerance = self.density / 2
        point_on_line = False
        for nv, start_point in enumerate(self.polygon[:-1]):
            end_point = self.polygon[nv + 1]
            direction_vector = end_point - start_point
            normal_vector = np.array([-direction_vector[1], direction_vector[0]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            third_point_new_rect = np.array(
                [(start_point[0] + tolerance * normal_vector[0]), (start_point[1] + tolerance * normal_vector[1])])
            fourth_point_new_rect = np.array(
                [(end_point[0] + tolerance * normal_vector[0]), (end_point[1] + tolerance * normal_vector[1])])
            check_polygon = np.array([start_point, end_point, fourth_point_new_rect, third_point_new_rect, start_point])
            is_on_line = self.check_vertice_polygon(point, check_polygon)
            if is_on_line:
                point_on_line = True
                break

        return point_on_line

    def get_seed_polygon(self):
        """
        Creates points inside polygon and on polygon outline
        Args:
        Nothing
        :return: np.array
        """

        all_outline_vertices_numbered, boundaries_numbered = self.create_polygon_outline_vertices()

        rect_min_max_coords = self.get_min_max_values()
        rect_seed_points = self.get_seed_rectangle(rect_min_max_coords)

        keep_points = []
        for idn, point in enumerate(rect_seed_points):
            if self.check_vertice(point):
                if not self.check_vertice_outline(point):
                    keep_points.append(idn)
        filtered_seed_points = rect_seed_points[keep_points]
        last_point_polygon_outline_vertice = all_outline_vertices_numbered[-1][0] + 1
        nbr_of_filtered_seed_points = len(filtered_seed_points)
        maxnode = last_point_polygon_outline_vertice + nbr_of_filtered_seed_points
        filtered_seed_points_numbered = [[n_point, filtered_seed_points[n_point - last_point_polygon_outline_vertice]]
                                         for n_point in range(last_point_polygon_outline_vertice, maxnode)]

        polygon_outline_vertices = [elem[1] for elem in all_outline_vertices_numbered]
        all_points = np.append(polygon_outline_vertices, filtered_seed_points, axis=0)
        all_points_numbered = all_outline_vertices_numbered + filtered_seed_points_numbered

        return all_points, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered

    def show_mesh(self):
        """

        :param all_points:
        :param polygon_outline_vertices:
        :param triangles:
        :return:
        """

        polygon_outline_vertices = np.array(self.polygon_outline_vertices)
        if not self.meshcreated:
            print(f"Run create_mesh() first!")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.scatter(polygon_outline_vertices[:, 0], polygon_outline_vertices[:, 1], c='b', marker='o',
                        label='Boundary Points')
            plt.scatter(self.all_points[:, 0], self.all_points[:, 1], c='b', marker='.', label='Seed Points')
            plt.triplot(self.all_points[:, 0], self.all_points[:, 1], self.triangles, c='gray', label='Mesh')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title('Mesh generation in Polygon')

            return fig, ax


    def output_mesh_param(self):
        """

        :return:
        """

        if len(self.all_points) < 1000:
            print("Node Coordinates:")
            for idp, point in enumerate(self.all_points):
                print(f"{idp}: {point}")

            print("Triangulation matrix:")
            for idt, triangle in enumerate(self.triangles):
                print(f"{idt}: {triangle}")
        else:
            print("Too many elements, printing would be unwise...")

    def write_output(self):
        """

        :return:
        """

        output_to_write = ''
        output_to_write += 'Coordinates of nodes\n'
        for idp, point in enumerate(self.all_points):
            output_to_write += f"{idp}: {point}\n"
        output_to_write += '\n\nTriangulation  matrix\n'
        for idt, triangle in enumerate(self.triangles):
            output_to_write += f"{idt}: {triangle}\n"
        with open('../output.txt', 'w') as f:
            f.write(output_to_write)

    def create_mesh(self):
        """

        :param polygon:
        :param density:
        :param method:
        :return:
        """

        all_points, all_points_numbered, all_outline_vertices_numbered, boundaries_numbered = self.get_seed_polygon()
        polygon_outline_vertices = [elem[1] for elem in all_outline_vertices_numbered]

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
            if self.check_vertice(center_point):
                keep_triangles.append(idt)
        triangles_filtered = triangles[keep_triangles]

        self.all_points = all_points
        self.polygon_outline_vertices = polygon_outline_vertices
        self.triangles = triangles_filtered
        self.meshcreated = True
        # self.output_mesh_param() # prints output

        return all_points_numbered, all_outline_vertices_numbered, boundaries_numbered, self.triangles