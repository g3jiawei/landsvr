import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def diff1(array):
    """
    :param array: input x
    :return: processed data which is the 1-order difference
    """
    return [j-i for i, j in zip(array[:-1], array[1:])]


def diff2(array):
    """
    :param array: input x
    :return: processed data which is the 2-order difference
    """
    return [j - i for i, j in zip(array[:-2], array[2:])]


def setup_equation(array, height):
    """
    :param array: input x
    :param height: input h
    :return: output y
    """
    diff_1 = diff1(array)
    diff_2 = diff2(array)
    # construct coefficients matrix and bias term
    para = np.zeros((len(height), len(height)))
    offset = np.zeros(len(height))
    para[0][0] = diff_2[0]
    para[0][1] = -diff_1[0]
    offset[0] = diff_2[0] * height[0]
    for i in range(1, len(height)-1):
        para[i, i-1] = -diff_2[i] + diff_1[i]
        para[i, i] = diff_2[i]
        para[i, i+1] = -diff_1[i]
        offset[i] = diff_2[i] * height[i]
    para[-1][-2] = -diff_1[-1]
    para[-1][-1] = diff_2[-1]
    offset[-1] = diff_2[-1] * height[-1]
    return para, offset

def main():
    # read from database
    df = pd.read_excel('dat.xlsx', 'Sheet1', header=None)
    x = df.as_matrix()
    df = pd.read_excel('dat.xlsx', 'Sheet2', header=None)
    h = df.as_matrix()
    # solve the equation
    A, b = setup_equation(x, h)
    result = np.linalg.solve(A, b)
    y = np.concatenate(([0], result, [0]))
    return x,y

if __name__ == '__main__':
    # read from database
    df = pd.read_excel('dat.xlsx', 'Sheet1', header=None)
    x = df.as_matrix()
    df = pd.read_excel('dat.xlsx', 'Sheet2', header=None)
    h = df.as_matrix()
    # solve the equation
    A, b = setup_equation(x, h)
    result = np.linalg.solve(A, b)
    y = np.concatenate(([0], result, [0]))
    #print(y)
    # plot the result
    #plt.plot(x, y)
    #plt.show()
