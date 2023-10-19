from scipy.interpolate import griddata
import numpy as np
from elements import ElementMatrices
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from typing import List, Tuple, Union
from scipy.sparse import coo_matrix


class CalcFEM:

    def __init__(self, all_points_numbered, all_outline_vertices_numbered,
                 boundaries_numbered, triangles, boundary_values, acoustic_source, corner_nodes):
        """
        Constructor
        :param all_points_numbered:
        :param all_outline_vertices_numbered:
        :param boundaries_numbered:
        :param triangles:
        :param boundary_values:
        """

        self.all_points_numbered = all_points_numbered
        self.all_outline_vertices_numbered = all_outline_vertices_numbered
        self.boundaries_numbered = boundaries_numbered
        self.triangles = triangles
        self.boundary_values = boundary_values
        self.acoustic_source = acoustic_source
        self.corner_nodes = corner_nodes

        self.all_points = np.array([elem[1] for elem in all_points_numbered])
        self.polygon_outline_vertices = np.array([elem[1] for elem in all_outline_vertices_numbered])
        self.k = 0.5
        self.zuordtab = triangles
        self.maxnode = len(self.all_points)
        self.maxelement = len(self.zuordtab)
        self.syssteifarray = None
        self.sysmassarray = None
        self.sysarray = None
        self.solution = None
        self.equation = 'WLG' # WLG for heat equation, 'HH' for Helmholtz equation
        self.freq = 1
        self.c_air = 340
        self.roh_air = 1.21

    def calculate_acoustic_source(self):
        if self.acoustic_source:
            omega = self.freq * 2 * math.pi
            mpsource = 4 * math.pi / self.roh_air * ((2 * self.roh_air * omega) / ((2 * math.pi) ** 2)) ** 0.5
            self.acoustic_source[-1] = mpsource

    def get_counter_clockwise_triangle(self, nodes: List[float]) -> List[float]:
        """
        rearanges the nodes, so that they are counter clockwise
        probably not necessary...
        :param nodes:
        :return:
        """

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        new_nodes = [nodes[0]]
        if x2 < x1:
            new_nodes.append(nodes[2])
            new_nodes.append(nodes[1])
        elif x2 == x1:
            if y2 < y1:
                new_nodes.append(nodes[1])
                new_nodes.append(nodes[2])
            else:
                new_nodes.append(nodes[2])
                new_nodes.append(nodes[1])
        elif x2 > x1:
            new_nodes.append(nodes[1])
            new_nodes.append(nodes[2])

        return new_nodes

    def calc_elementmatrices(self):
        """

        :return:
        """

        self.nbr_of_elements = len(self.triangles)

        self.all_element_matrices_steif = np.zeros((self.nbr_of_elements, 3, 3), dtype=np.single)
        self.all_element_matrices_mass = np.zeros((self.nbr_of_elements, 3, 3), dtype=np.single)

        for idx, triangle in enumerate(self.triangles):
            p1, p2, p3 = triangle[0], triangle[1], triangle[2]
            x1 = self.all_points[p1][0]
            y1 = self.all_points[p1][1]
            x2 = self.all_points[p2][0]
            y2 = self.all_points[p2][1]
            x3 = self.all_points[p3][0]
            y3 = self.all_points[p3][1]
            nodes = [[x1, y1], [x2, y2], [x3, y3]]
            if self.equation == 'WLG':
                elemsteif, elemmass = ElementMatrices.calc_2d_triangular_heatflow_p1(self.k, nodes)
            elif self.equation == 'HH':
                elemsteif, elemmass = ElementMatrices.calc_2d_triangular_acoustic_p1(nodes)
            self.all_element_matrices_steif[idx] = elemsteif
            if elemmass is not None: # # since it might be a np.array
                self.all_element_matrices_mass[idx] = elemmass

    def calc_force_vector(self):
        """

        :return:
        """

        self.lastvektor = np.zeros(self.maxnode, dtype=np.single)
        if self.acoustic_source:
            pos = self.acoustic_source[0][0]
            value = self.acoustic_source[-1]
            self.lastvektor[pos] = self.lastvektor[pos] + value
            print("xxx")
            print(self.acoustic_source)
            print(self.lastvektor)

    def calc_system_matrices(self):
        """

        :return:
        """

        self.syssteifarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)
        self.sysmassarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)
        self.sysarray = np.zeros((self.maxnode, self.maxnode), dtype=np.single)

        for ielem in range(self.nbr_of_elements):
            elesteifmat = self.all_element_matrices_steif[ielem]
            elemassmat = self.all_element_matrices_mass[ielem]
            for a in range(3):
                for b in range(3):
                    zta = int(self.zuordtab[ielem, a])
                    ztb = int(self.zuordtab[ielem, b])
                    self.syssteifarray[zta, ztb] = self.syssteifarray[zta, ztb] + elesteifmat[a, b]
                    self.sysmassarray[zta, ztb] = self.sysmassarray[zta, ztb] + elemassmat[a, b]

        if self.equation == 'WLG':
            self.sysarray = self.syssteifarray
        elif self.equation == 'HH':  # todo: include in elementmatrices
            omega = self.freq * 2 * math.pi
            self.sysarray = (1/self.roh_air) * self.syssteifarray \
                            - omega**2 * (1/self.roh_air * 1/(self.c_air**2)) * self.sysmassarray

    def calc_system_matrices_new(self):
        """
        todo: implement more efficient method for assembly
        :return:
        """

        element_stiff_array = np.asarray(self.all_element_matrices_steif)
        element_mass_array = np.asarray(self.all_element_matrices_mass)

        vectora = [[[int(self.zuordtab[ielem, a]) for b in range(3)] for a in range(3)]
                   for ielem in range(self.maxelement)]
        vectora = np.asarray(vectora)
        vectora = vectora.flatten()
        row = vectora

        vectorb = [[[int(self.zuordtab[ielem, b]) for b in range(3)] for a in range(3)]
                   for ielem in range(self.maxelement)]
        vectorb = np.asarray(vectorb)
        vectorb = vectorb.flatten()
        col = vectorb

        data_stiff = element_stiff_array.flatten()
        data_mass = element_mass_array.flatten()
        syssteifarraycoo = coo_matrix((data_stiff, (row, col)), shape=(self.maxnode, self.maxnode))
        sysmassarraycoo = coo_matrix((data_mass, (row, col)), shape=(self.maxnode, self.maxnode))

        if self.equation == 'WLG':
            self.sysarray = syssteifarraycoo
        elif self.equation == 'HH':
            omega = self.freq * 2 * math.pi
            self.sysarray = syssteifarraycoo - omega**2 * sysmassarraycoo



    def implement_diriclet(self, sysmatrix, forcevector, diriclet_list):
        """

        :param sysmatrix:
        :param forcevector:
        :param diriclet_list:
        :return:
        """

        diriclet_list_positions = [pos[0] for pos in diriclet_list]
        diriclet_list_values = [pos[1] for pos in diriclet_list]
        original_sysmatrix = np.copy(sysmatrix)

        # sysmatrix
        for position, value in diriclet_list:
            sysmatrix[:, position] = 0
            sysmatrix[position, :] = 0
            sysmatrix[position, position] = 1

        # force vector
        forcevector = forcevector - np.dot(original_sysmatrix[:, diriclet_list_positions], diriclet_list_values)
        for pos, value in diriclet_list:
            forcevector[pos] = value

        return sysmatrix, forcevector

    def print_matrix(self, matrix: Union[np.array, list]):
        """
        Prints matrix
        :param matrix: [[val1, val2, val3],[val4,...],[...]], either np.array or list
        """

        if not isinstance(matrix[0], np.ndarray):
            if len(matrix) < 50:
                print("[", end='')
                for idx, val in enumerate(matrix):
                    if idx < len(matrix) - 1:
                        print(f"+{abs(val):.2f}," if val >= 0 else f"-{abs(val):.2f},", end='')
                    else:
                        print(f"+{abs(val):.2f}" if val >= 0 else f"-{abs(val):.2f}", end='')
                print("]")

        else:
            if len(matrix) < 50:
                print("[", end='\n')
                for idx, elem in enumerate(matrix):
                    print("[", end='')
                    for idy, val in enumerate(elem):
                        if val == 0:
                            val_str = '_____'
                        else:
                            val_str = f"+{abs(val):.2f}" if val >= 0 else f"-{abs(val):.2f}"
                        if idy < len(elem) - 1:
                            print(f"{val_str},", end='')
                        elif idy >= len(elem) - 1 and idx < len(matrix) - 1:
                            print(f"{val_str}],", end='\n')
                        else:
                            print(f"{val_str}]", end='\n')
                print("]")

    def solve_linear_system(self):
        """
        Solves the linear system
        """

        if self.sysarray is not None:  # since it might be a np.array
            self.solution = np.linalg.solve(self.sysarray, self.lastvektor)

    def plot_solution_dev(self):
        """
        Plots the solution via matplotlib
        """

        plot_label_dict = {'WLG': {'title': 'Temperature Field', 'ylabel': 'Temp [K]'},
                           'HH': {'title': 'Pressure Field', 'ylabel': 'P [pa]'}}
        dataz = np.real(self.solution)
        values = dataz
        aspectxy = 1
        triang_mpl = tri.Triangulation(self.all_points[:, 0], self.all_points[:, 1], self.triangles)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(plot_label_dict[self.equation]['title'])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect(aspectxy)

        contour = ax.tricontourf(triang_mpl, values, cmap='viridis', levels=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(contour, cax=cax)
        cbar.ax.set_ylabel(plot_label_dict[self.equation]['ylabel'])
        ax.triplot(triang_mpl, 'w-', linewidth=0.1)

        plt.show()


    def plot_solution(self):
        """
        Plots the solution via matplotlib
        """

        plot_label_dict = {'WLG': {'title': 'Temperature Field', 'ylabel': 'Temp [K]'},
                           'HH': {'title': 'Pressure Field', 'ylabel': 'P [pa]'}}
        dataz = np.real(self.solution)
        values = dataz
        aspectxy = 1
        triang_mpl = tri.Triangulation(self.all_points[:, 0], self.all_points[:, 1], self.triangles)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_title(plot_label_dict[self.equation]['title'])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect(aspectxy)

        contour = ax.tricontourf(triang_mpl, values, cmap='viridis', levels=20)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(contour, cax=cax)
        cbar.ax.set_ylabel(plot_label_dict[self.equation]['ylabel'])

        ax.scatter(self.all_points[:, 0], self.all_points[:, 1], c=values, cmap='viridis', marker='.',
                             edgecolors='w', s=10)
        ax.triplot(triang_mpl, 'w-', linewidth=0.1)

        return fig, ax

    def create_impedance_matrix(self):
        ...

    def calc_fem(self):
        """
        main method for calculation, calculates elementmatrices, assembly of system matrices,
        implements boundary conditions, solves linear system
        :return:
        """

        all_diriclet = []
        for boundary_number, boundary_value in self.boundary_values.items():
            this_boundary = self.boundaries_numbered[boundary_number]
            boundary_node_numbers_value = [(nbr[0], boundary_value[0]) for nbr in this_boundary if boundary_value[1] == 'Dirichlet']
            all_diriclet += boundary_node_numbers_value

        self.calc_elementmatrices()
        self.calc_system_matrices()
        ###
        # Test für effizientere Assemblierung
        # self.calc_system_matrices_new()
        ###
        self.calculate_acoustic_source()
        self.calc_force_vector()

        # implement neumann boundary conditions
        self.create_impedance_matrix()

        # Implementing dirichlet boundary conditions
        sysmatrix_adj, force_vector_adj = self.implement_diriclet(self.sysarray, self.lastvektor, all_diriclet)
        self.sysarray = sysmatrix_adj
        self.lastvektor = force_vector_adj
        self.solve_linear_system()

        return self.solution


if __name__ == "__main__":
    ...  # todo: example input and output
