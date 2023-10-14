import numpy as np

def jacobi(nodes: np.array, xi1: float, xi2: float):
    """
    jacobi matrixfor triangular element
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

nodes = np.array([[0,0],[1.1,0.1],[0.5,0.6]])
print(jacobi(nodes, 0.12,1.3))

