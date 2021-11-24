import numpy as np


def calculate(data):
    data = np.array(data)
    print('mean: {}, var: {}, std: {}'.format(data.mean(), data.var(), data.std()))


if __name__ == '__main__':
    calculate([917, 835, 872])
