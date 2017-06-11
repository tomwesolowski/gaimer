import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

WSCALE, HSCALE = 1, 1
INF = 1e9

class PlotType(Enum):
    REGULAR = 0
    BAR = 1
    MATRIX = 2


class Gui:
    def __init__(self, height, width, epochs):
        self.height = height
        self.width = width
        self.epochs = epochs

    def run(self, colorboard, stats):
        plt.ion()
        self.fig = plt.gcf()
        self.fig.set_size_inches(self.height*HSCALE, self.width*WSCALE, forward=True)
        self.axboard = self.fig.add_axes([0, 0.3, 1, 0.7], frameon=True)
        self.mat = self.axboard.matshow(np.zeros((self.height, self.width)), vmin=0, vmax=255)
        self.mat.set_data(colorboard)

        self.stats = dict()
        statid = 0
        for (name, stat) in stats.items():
            ax = self.fig.add_axes([0.05 + statid * 0.3, 0.05, 0.3, 0.2], frameon=True)
            if stat.plot_type == PlotType.REGULAR:
                line = ax.plot(0, 0)[0]
            elif stat.plot_type == PlotType.BAR:
                line = ax.bar(0, 0)[0]
            elif stat.plot_type == PlotType.MATRIX:
                line = ax.matshow(stat.matrix, vmin=-100, vmax=0)
            else:
                assert False
            statid += 1
            ax.set_title(name)
            self.stats[name] = (ax, line)
        self.refresh()

    def update_stats(self, stats):
        for (name, stat) in stats.items():
            if stat.plot_type == PlotType.REGULAR:
                self.update_regular(name, stat)
            elif stat.plot_type == PlotType.BAR:
                self.update_bar(name, stat)
            elif stat.plot_type == PlotType.MATRIX:
                self.update_matrix(name, stat)
            else:
                assert False
        self.refresh()

    def update_regular(self, name, stat):
        ax, line = self.stats[name]
        line.set_xdata(range(len(stat.values)))
        line.set_ydata(stat.values)
        ax.set_xlim(0, len(stat.values))
        ax.set_ylim(min(stat.values), max(stat.values))

    def update_bar(self, name, stat):
        ax, _ = self.stats[name]
        ax.clear()
        xlen = len(stat.values)
        if xlen <= 0:
            return
        X = range(xlen)
        Y = np.array(stat.values)[:, 0]
        L = np.array(stat.values)[:, 1]
        rects = ax.bar(X, Y)
        for x, y, l in zip(X, Y, L):
            plt.text(x, y, str(l), ha='center', va='top')
        self.stats[name] = (ax, rects)

    def update_matrix(self, name, stat):
        (ax, mat) = self.stats[name]
        mat.set_data(stat.matrix)

    def refresh(self):
        self.fig.canvas.draw()
        plt.show()
        plt.pause(0.02)

    def update_board(self, colorboard):
        self.mat.set_data(colorboard)
        self.fig.canvas.draw()
