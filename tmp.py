import numpy as np

mat = [[11,12,13,14,15],
       [21,22,23,24,25],
       [31,32,33,34,35],
       [41,42,43,44,45],
       [51,52,53,54,55]]




def implement_diriclet(sysmatrix, diriclet_list):
    for position, value in diriclet_list:
        sysmatrix[:, position] = 0
        sysmatrix[position, :] = 0
        sysmatrix[position, position] = 1

    return sysmatrix

sysmatrix = np.array(mat)
print(sysmatrix)
diriclet_list = [(0, 22), (2, 33)]
sysmatrix_dc = implement_diriclet(sysmatrix, diriclet_list)
print(sysmatrix_dc)

