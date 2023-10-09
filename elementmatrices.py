import numpy as np


##########################
# 2D Triangular heatflow element
# Corners counter clockwise
#

class ElementMatrice:

    def __init__(self):
        ...

    @staticmethod
    def calc_2d_triangulat_heatflow(conductivity: float, nodes: list):
        """

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


k = 0.5
nodes = np.array([[0.2, 0.1], [1.1, 0.3], [0.5, 0.55]])
kmat = ElementMatrice.calc_2d_triangulat_heatflow(k, nodes)
print(kmat)
