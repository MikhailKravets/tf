import csv
import numpy as np


def from_csv(file_name: str) -> (np.ndarray, np.ndarray):
    """
    Load the data from specially formatted .csv file into list of tuples which contain vectors (input data
    and desired output).

    :param file_name: path to `.csv` file
    :return: tuple of X and Y numpy arrays
    """
    x, y = [], []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                rx = list(map(lambda e: float(e), row[:row.index('-')]))
                ry = list(map(lambda e: float(e), row[row.index('-') + 1:]))
                x.append(rx)
                y.append(ry)
            except ValueError:
                pass
    x, y = np.array(x), np.array(y)
    return x, y


def separate_data(x: np.ndarray, y: np.ndarray, percentage: float) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
    """
    Separate the data into training set and validation set with the given percentage

    :param x: list of vectors - input data
    :param y: list of vectors - desired output
    :param percentage: float value between [0, 1) which defines how much data move to validation set
    :return: tuple of training set, validation set
    """
    v_len = int(percentage * len(x))
    vx, vy = [], []
    tx, ty = [], []
    indices = np.random.choice(len(x), v_len)

    for i in range(len(x)):
        if i in indices:
            vx.append(x[i])
            vy.append(y[i])
        else:
            tx.append(x[i])
            ty.append(y[i])

    vx, vy = np.array(vx), np.array(vy)
    tx, ty = np.array(tx), np.array(ty)
    return tx, ty, vx, vy


if __name__ == '__main__':
    x, y = from_csv("D:\\DELETE\\Дипломмо\\output.csv")
    tx, ty, vx, vy = separate_data(x, y, 0.15)
