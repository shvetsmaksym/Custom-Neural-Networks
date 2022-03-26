import matplotlib.pyplot as plt
import numpy as np

from DataStructures.Neurons import Neuron1


def nor_network_architecture(x1, x2):
    """Zwraca wynik sieci NOR"""
    n1 = Neuron1(x=[x1, x2], weights=[1, 1], thresh=1)
    y_a = n1.calculate()

    n2 = Neuron1(x=[y_a], weights=[-1], thresh=0)
    y = n2.calculate()

    return y


def nand_network_architecture(x1, x2):
    """Zwraca wynik sieci NAND"""
    n10 = Neuron1(x=[x1], weights=[-1], thresh=0)
    y10 = n10.calculate()
    n20 = Neuron1(x=[x2], weights=[-1], thresh=0)
    y20 = n20.calculate()

    n11 = Neuron1(x=[y10, y20], weights=[1, 1], thresh=1)
    y = n11.calculate()

    return y


def calculate_results_for_nor():
    """Tworzy listę współrzędnych dla sieci NOR oraz dopisuje wynik obliczeń dokonanych przez sieć"""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    X[0].append(nor_network_architecture(X[0][0], X[0][1]))
    X[1].append(nor_network_architecture(X[1][0], X[1][1]))
    X[2].append(nor_network_architecture(X[2][0], X[2][1]))
    X[3].append(nor_network_architecture(X[3][0], X[3][1]))

    return X


def calculate_results_for_nand():
    """Tworzy listę współrzędnych dla sieci NAND oraz dopisuje wynik obliczeń dokonanych przez sieć"""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    X[0].append(nand_network_architecture(X[0][0], X[0][1]))
    X[1].append(nand_network_architecture(X[1][0], X[1][1]))
    X[2].append(nand_network_architecture(X[2][0], X[2][1]))
    X[3].append(nand_network_architecture(X[3][0], X[3][1]))

    return X


def plot_results(X, title='title'):
    """Rysuje punkty na podstawie specjalnej listy współrzędnych obliczonych przez którąś z powyższych dwóch funkcji"""
    xpoints_0 = np.array([x[0] for x in X if x[2] == 0])
    ypoints_0 = np.array([x[1] for x in X if x[2] == 0])

    xpoints_1 = np.array([x[0] for x in X if x[2] == 1])
    ypoints_1 = np.array([x[1] for x in X if x[2] == 1])

    plt.plot(xpoints_0, ypoints_0, 'o', color='blue', markersize=12)
    plt.plot(xpoints_1, ypoints_1, 'o', color='red', markersize=12)
    plt.xlabel("x1", color='black', size=15)
    plt.ylabel("x2", color='black', size=15)
    plt.title(title, color='black', size=15)
    plt.show()


if __name__ == '__main__':
    # Współrzędne
    xy_nor = calculate_results_for_nor()
    xy_nand = calculate_results_for_nand()

    # Wykresy
    plot_results(xy_nor, "NOR")
    plot_results(xy_nand, "NAND")
