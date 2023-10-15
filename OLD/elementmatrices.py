import numpy as np


##########################
# 2D Triangular heatflow element
# Corners counter clockwise
#

# TODO: Vermutlich stimmen Jacobimatrizen nicht, da falsche Formfunktionen bei Berechnung verwendet -____-

class ElementMatrice:

    def __init__(self):
        ...

    @staticmethod
    def jacobi(nodes: np.array, xi1: float, xi2: float):
        """
        jacobi matrix for triangular element
        :param nodes: np.array [[x1, y1],[x2, y2],[x3,y3]]
        :return:
        """
        x1 = nodes[0, 0]
        y1 = nodes[0, 1]
        x2 = nodes[1, 0]
        y2 = nodes[1, 1]
        x3 = nodes[2, 0]
        y3 = nodes[2, 1]

        j11 = x2 * (1. - 1. * xi2) - 1. * x3 * xi2 + x1 * (-1. + 1. * xi2)
        j12 = x3 * (1. - 1. * xi1) - 1. * x2 * xi1 + x1 * (-1. + 1. * xi1)
        j21 = (-1. + 1. * xi2) * y1 + (1. - 1. * xi2) * y2 - 1. * xi2 * y3
        j22 = (-1. + 1. * xi1) * y1 - 1. * xi1 * y2 + 1. * y3 - 1. * xi1 * y3

        return np.array([[j11, j12], [j21, j22]])


    @staticmethod
    def jacobi_det(nodes: np.array, xi1: float, xi2: float):
        """
        jacobi determinant for triangular element
        :param nodes: np.array [[x1, y1],[x2, y2],[x3,y3]]
        :return:
        """
        x1 = nodes[0, 0]
        y1 = nodes[0, 1]
        x2 = nodes[1, 0]
        y2 = nodes[1, 1]
        x3 = nodes[2, 0]
        y3 = nodes[2, 1]

        det = x3 * (1. - 1. * xi1) * y1 + x2 * (-1. + 1. * xi2) * y1 \
              + 1. * x1 * y2 - 1. * x1 * xi2 * y2 + x3 * (-1. + 1. * xi1 + 1. * xi2) * y2 \
              + x1 * (-1. + 1. * xi1) * y3 + x2 * (1. - 1. * xi1 - 1. * xi2) * y3

        return det


    @staticmethod
    def jacobi_inv_trans(nodes: np.array, xi1: float, xi2: float):
        """
        Inverse(Transpose(Jacobi)) for triangular element
        :param nodes: np.array [[x1, y1],[x2, y2],[x3,y3]]
        :return:
        """
        x1 = nodes[0, 0]
        y1 = nodes[0, 1]
        x2 = nodes[1, 0]
        y2 = nodes[1, 1]
        x3 = nodes[2, 0]
        y3 = nodes[2, 1]

        jit11 = ((1. - 1. * xi1) * y1 + 1. * xi1 * y2 - 1. * y3 + 1. * xi1 * y3) / (
                    0. + x3 * (-1. + 1. * xi1) * y1 + x2 * (1. - 1. * xi2) * y1 - 1. * x1 * y2 + x3 * (
                        1. - 1. * xi1 - 1. * xi2) * y2 + 1. * x1 * xi2 * y2 + x1 * (1. - 1. * xi1) * y3 + x2 * (
                                -1. + 1. * xi1 + 1. * xi2) * y3)
        jit12 = ((1. - 1. * xi2) * y1 + (-1. + 1. * xi2) * y2 + 1. * xi2 * y3) / (
                    0. + x3 * (1. - 1. * xi1) * y1 + x2 * (
                        -1. + 1. * xi2) * y1 + 1. * x1 * y2 - 1. * x1 * xi2 * y2 + x3 * (
                                -1. + 1. * xi1 + 1. * xi2) * y2 + x1 * (-1. + 1. * xi1) * y3 + x2 * (
                                1. - 1. * xi1 - 1. * xi2) * y3)
        jit21 = (x1 * (1. - 1. * xi1) + 1. * x2 * xi1 + x3 * (-1. + 1. * xi1)) / (
                    0. + x3 * (1. - 1. * xi1) * y1 + x2 * (
                        -1. + 1. * xi2) * y1 + 1. * x1 * y2 - 1. * x1 * xi2 * y2 + x3 * (
                                -1. + 1. * xi1 + 1. * xi2) * y2 + x1 * (-1. + 1. * xi1) * y3 + x2 * (
                                1. - 1. * xi1 - 1. * xi2) * y3)
        jit22 = (x2 * (1. - 1. * xi2) - 1. * x3 * xi2 + x1 * (-1. + 1. * xi2)) / (
                    0. + x3 * (1. - 1. * xi1) * y1 + x2 * (
                        -1. + 1. * xi2) * y1 + 1. * x1 * y2 - 1. * x1 * xi2 * y2 + x3 * (
                                -1. + 1. * xi1 + 1. * xi2) * y2 + x1 * (-1. + 1. * xi1) * y3 + x2 * (
                                1. - 1. * xi1 - 1. * xi2) * y3)

        return np.array([[jit11, jit12], [jit21, jit22]])


    @staticmethod
    def calc_2d_triangulat_heatflow(conductivity: float, nodes: list):
        """
        precalculated element
        :param conductivity: k
        :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        :return: np.array
        """

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]
        k = conductivity

        val11 = -((k * (x2 - x3 - y2 + y3) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val12 = (k * (-x2 + x3 + y2 - y3) * (x1 - x3 - y1 + y3)) / (
                    2 * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)))
        val13 = -((k * (x1 - x2 - y1 + y2) * (x2 - x3 - y2 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val21 = val12
        val22 = -((k * (x1 - x3 - y1 + y3) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        val23 = (k * (x1 - x2 - y1 + y2) * (x1 - x3 - y1 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3)))
        val31 = val13
        val32 = val23
        val33 = -((k * (x1 - x2 - y1 + y2) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y1 - y3) + x1 * (-y2 + y3))))
        kmat = np.array([[val11, val12, val13], [val21, val22, val23], [val31, val32, val33]])

        return kmat


    @staticmethod
    def calc_2d_triangulat_heatflow_new(conductivity: float, nodes: list):
        """
        precalculated element
        :param conductivity: k
        :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        :return: np.array
        """

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]
        k = conductivity

        val11 = -((x2 - x3) ** 2 + (y1 - 2 * y2 + y3) ** 2) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val12 = ((x1 - x3) * (x2 - x3) - (y2 - y3) * (y1 - 2 * y2 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val13 = (-((x1 - x2) * (x2 - x3)) + (y1 - y2) * (y1 - 2 * y2 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val21 = ((x1 - x3) * (x2 - x3) - (y2 - y3) * (y1 - 2 * y2 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val22 = -((x1 - x3) ** 2 + (y2 - y3) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val23 = (x1 ** 2 + x2 * x3 - x1 * (x2 + x3) + (y1 - y2) * (y2 - y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val31 = (-((x1 - x2) * (x2 - x3)) + (y1 - y2) * (y1 - 2 * y2 + y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val32 = (x1 ** 2 + x2 * x3 - x1 * (x2 + x3) + (y1 - y2) * (y2 - y3)) / (
                    2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        val33 = -((x1 - x2) ** 2 + (y1 - y2) ** 2) / (2 * (x3 * (-y1 + y2) + x2 * (y2 - y3) + x1 * (y1 - 2 * y2 + y3)))
        kmat = np.array([[val11, val12, val13], [val21, val22, val23], [val31, val32, val33]])

        return kmat


    @staticmethod
    def calc_2d_triangulat_heatflow_order(conductivity: float, nodes: list, order: int):
        """
        calculation for various orders
        :param conductivity: k
        :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
        :return: np.array
        """

        x1 = nodes[0][0]
        y1 = nodes[0][1]
        x2 = nodes[1][0]
        y2 = nodes[1][1]
        x3 = nodes[2][0]
        y3 = nodes[2][1]
        k = conductivity

        xi1 = 10.12 # todo, dummy
        xi2 = 11.3 # todo, dummy


        # NMAT IST FALSCH TODO TODO TODO
        # nmat = [(1.0 - 1.0 * xi1) * (1.0 - 1.0 * xi2), xi1 * (1.0 - 1.0 * xi2), (1.0 - 1.0 * xi1) * xi2] # Informell
        # nable = [[d/d_xi1],[d/d_xi2]] # Informell


        # TODO: Formfunktionen stimmen nicht!!!
        nabla_nmat_11 = -1.*(1. - 1.*xi2)
        nabla_nmat_12 = 1. - 1.*xi2
        nabla_nmat_13 = -1.*xi2
        nabla_nmat_21 = -1.*(1. - 1.*xi1)
        nabla_nmat_22 = -1.*xi1
        nabla_nmat_23 = 1. - 1.*xi1
        nabla_mat = np.array([[nabla_nmat_11, nabla_nmat_12, nabla_nmat_13],
                              [nabla_nmat_21, nabla_nmat_22, nabla_nmat_23]])

        nodes_np = np.array(nodes)
        jac_inv_trans = ElementMatrice.jacobi_inv_trans(nodes_np, xi1, xi2)

        bmat = np.dot(jac_inv_trans, nabla_mat)
        bmat_trans = np.transpose(bmat)

        integrand = np.dot(bmat_trans, bmat)

        print(nabla_mat)


        return None


k = 0.5
nodes = np.array([[0,0],[1.1,0.1],[0.5,0.6]])
kmat = ElementMatrice.calc_2d_triangulat_heatflow(k, nodes)
kmat_calc = ElementMatrice.calc_2d_triangulat_heatflow_order(k, nodes, 1)
