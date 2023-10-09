import numpy as np
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
from numpy import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Numerische Integration

numintgld1 = np.array(
    [[0.21132486, 0.78867513], [0.50000000, 0.50000000]])  # Numerische Integration 1D Knoten und Gewichte
numintgld2nodes = np.array([[0.21132487, 0.21132487], [0.21132487, 0.78867513], \
                            [0.78867513, 0.21132487], [0.78867513, 0.78867513]])  # Numerische Integration 2D Knoten
numintgld2weights = np.array([0.25000000, 0.25000000, 0.25000000, 0.25000000])  # Numerische Integration 2D Gewichte


# --------------------------------------------------
# Determinante der Jacobimatrix für die Eckpunkte {{x1, y1}, {x2, y2}, {x3, y3}, {x4,y4}}
# -> {{ul}, {ur}, {ol}, {or}} mit den Funktionswerten xi1 und xi2
def jacdetc(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4):
    f = 0.0 - 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
        1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - \
        1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + 1.0 * x2 * y3 + \
        1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - \
        1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4
    return f


# --------------------------------------------------
# Formfunktionen philequdist und phiqequdist
def philequdist(knoten, xi):
    knoten = int(knoten)
    if knoten == 1:
        f = 1.0 - xi
    elif knoten == 2:
        f = xi
    return f


def phiqequdist(knoten, xi1, xi2):
    knoten = int(knoten)
    if knoten == 1:
        f = (1 - xi1) * (1 - xi2)
    elif knoten == 2:
        f = xi1 * (1 - xi2)
    elif knoten == 3:
        f = (1 - xi1) * xi2
    elif knoten == 4:
        f = xi1 * xi2
    return f


def phiqequdistarray(xi1, xi2):
    f = np.array([[(1 - xi1) * (1 - xi2)], [xi1 * (1 - xi2)], [(1 - xi1) * xi2], [xi1 * xi2]])
    return f


# --------------------------------------------------
# phiqequdistgrad
def phiqequdistgrad(knoten, xi1, xi2):
    knoten = int(knoten)
    if knoten == 1:
        f = np.array([-1 + xi2, -1 + xi1])
    elif knoten == 2:
        f = np.array([1 - xi2, -xi1])
    elif knoten == 3:
        f = np.array([-xi2, 1 - xi1])
    elif knoten == 4:
        f = np.array([xi2, xi1])
    return f


# --------------------------------------------------
# gradmatequdist
def gradmatequdist(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4):
    jacinvtrans = np.array(
        [[(-1.0 * (1.0 - 1.0 * xi1) * y1 - 1.0 * xi1 * y2 + (1.0 - 1.0 * xi1) * y3 + xi1 * y4) / (0.0 - \
                                                                                                  1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
                                                                                                  1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + \
                                                                                                  1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - \
                                                                                                  1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - \
                                                                                                  1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + \
                                                                                                  1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4), \
          (1.0 * (1.0 - 1.0 * xi2) * y1 - (1.0 - 1.0 * xi2) * y2 + \
           1.0 * xi2 * y3 - xi2 * y4) / (0.0 - 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + \
                                         1.0 * x4 * xi1 * y1 + 1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - \
                                         1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + \
                                         1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - \
                                         1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + \
                                         1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4)], \
         [(1.0 * x1 * (1.0 - 1.0 * xi1) - x3 * (1.0 - 1.0 * xi1) + 1.0 * x2 * xi1 - x4 * xi1) / (0.0 - \
                                                                                                 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
                                                                                                 1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + \
                                                                                                 1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - \
                                                                                                 1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - \
                                                                                                 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + \
                                                                                                 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4), \
          (-1.0 * x1 * (1.0 - 1.0 * xi2) + \
           x2 * (1.0 - 1.0 * xi2) - 1.0 * x3 * xi2 + x4 * xi2) / (0.0 - 1.0 * x2 * y1 + \
                                                                  1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + 1.0 * x2 * xi2 * y1 - \
                                                                  1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - \
                                                                  1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + \
                                                                  1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + \
                                                                  1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - \
                                                                  1.0 * x3 * xi2 * y4)]])

    vekgradmat = np.zeros((4, 2))

    for knoten in range(4):
        matrow = jacinvtrans @ phiqequdistgrad(knoten + 1, xi1, xi2)
        vekgradmat[knoten] = matrow
    return vekgradmat


intnodes=numintgld2nodes
intweights=numintgld2weights

elesteifmat = np.zeros((4, 4))
elemassenmat = np.zeros((4, 4))

x1 = 1
y1 = 1
x2 = 2.5
y2 = 1.2
x3 = 2.4
y3 = 2.0
x4 = x3
y4 = y3

for i in range(4):
    xi1 = intnodes[i, 0]
    xi2 = intnodes[i, 1]
    phi = phiqequdistarray(xi1, xi2)
    phit = np.transpose(phiqequdistarray(xi1, xi2))
    phixphit = phi @ phit
    jd = jacdetc(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4)
    fp = phixphit * jd * intweights
    elemassenmat = elemassenmat + fp

    for i in range(4):
        xi1=intnodes[i,0]
        xi2=intnodes[i,1]
        gr=gradmatequdist(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4)
        grt=np.transpose(gr)
        grxgrt=gr@grt
        jd=jacdetc(xi1,xi2,x1,x2,x3,x4,y1,y2,y3,y4)
        fp=grxgrt*jd*intweights
        elesteifmat=elesteifmat+fp
    elesteifmat=elesteifmat

elemassenmat = [[]]

print(elemassenmat)
print(elesteifmat)

