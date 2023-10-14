import numpy as np

def calc_2d_triangulat_heatflow_new(conductivity: float, nodes: list):

    """
            :param nodes: [[x1, y1],[x2, y2],[x3, y3]]
    """
    def n1(xi1, xi2):
        return 1 - xi1 - xi2

    def n2(xi1, xi2):
        return xi1

    def n3(xi1, xi2):
        return xi2

    def ngrad1(xi1, xi2):
        return np.array([-1, -1])

    def ngrad2(xi1, xi2):
        return np.array([1, 0])

    def ngrad3(xi1, xi2):
        return np.array([0, 1])

    def gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3):
        jacobi_inverse_transpose_matrix = np.array([[(y1 - y3) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3),
                                            (y1 - y2) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))],
                                           [(x1 - x3) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3)),
                                            (x1 - x2) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3)]], dtype=np.single)

        ngrad = np.array([ngrad1(xi1, xi2), ngrad2(xi1, xi2),
                          ngrad3(xi1, xi2)], dtype=np.single)

        return np.transpose(np.dot(jacobi_inverse_transpose_matrix, np.transpose(ngrad)))

    x1 = nodes[0][0]
    y1 = nodes[0][1]
    x2 = nodes[1][0]
    y2 = nodes[1][1]
    x3 = nodes[2][0]
    y3 = nodes[2][1]
    k = conductivity


    intnodes = np.array([[0, 0], [1, 0], [0, 1]])
    intweights = np.array([1/6, 1/6, 1/6])

    jacobi_inverse_transp_11 = (y1 - y3) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3)
    jacobi_inverse_transp_12 = (y1 - y2) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))
    jacobi_inverse_transp_21 = (x1 - x3) / (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (-y1 + y3))
    jacobi_inverse_transp_22 = (x1 - x2) / (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3)

    jacobi_det = -x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3

    elesteifmat = np.zeros((3, 3), dtype=np.single)
    for i in range(3):
        xi1 = intnodes[i, 0]
        xi2 = intnodes[i, 1]
        gr = gradmat(xi1, xi2, x1, x2, x3, y1, y2, y3)
        grt = np.transpose(gr)
        grxgrt = gr @ grt
        fp = grxgrt * jacobi_det * intweights
        elesteifmat = elesteifmat + fp

    return elesteifmat

res = calc_2d_triangulat_heatflow_new(1, [[0.1, 0.1], [1.1, 0], [0, 0.9]])
print(res)

