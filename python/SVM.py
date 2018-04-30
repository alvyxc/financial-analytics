import numpy as np
import pandas as pd


def get_gaussian_matrix(x, sigma):
    row,col = x.shape
    gass_matrix = np.zeros(shape=(row,row))
    i = 0
    for vi in x:
        j = 0
        for vj in x:
            gass_matrix[i,j] = gaussian_kernel(vi.T, vj.T, sigma)
            j += 1
        i += 1
    return gass_matrix


def gaussian_kernel(x, z, sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))

t = np.matrix([[2.5, 1], [3.5, 4], [2, 2.1]])
result = get_gaussian_matrix(t, np.sqrt(5))


print result
eigvalue = np.linalg.eig(result)
print eigvalue
