from datetime import timedelta
import sys
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .stats import acf

plt.style.use('ggplot')


class Plots:

    def trace_plots(
        self,
        show: bool = True,
        save: bool = False,
        name: str = 'traces'
    ) -> None:
        """Generate the traceplots of the posterior samples of the para-
        meters of interest. The plots can be displayed in one single
        figure on the screen and/or saved as a picture. A custom name
        can be given to the saved file.

        Args:
            self (object): A class instance that has attributes
                'fullchain', 'nonspat' and '_names' defined.
            show (bool, optional): Whether to show the plot on screen.
                Defaults to True.
            save (bool, optional): Whether to save plot as a picture.
                Defaults to False.
            name (str, optional): Name of the picture if save = True.
                The supplied input can also be a path if it is to be
                saved in a custom directory. Defaults to 'traces'.
        """
        traces = self.fullchain
        if self.nonspat:
            plot_rows = traces.shape[1] - 1
        else:
            plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            plt.subplot(plot_rows, 2, 2*i + 1)
            plt.plot(data)
            plt.title(self._names[i])
            plt.subplot(plot_rows, 2, 2*i + 2)

            s_data = sorted(data)
            plt.plot(s_data, gaussian_kde(data).pdf(s_data), linewidth=2)
            plt.hist(
                data,
                bins=55,
                density=True,
                histtype='stepfilled',
                color='red',
                alpha=0.3
            )
            plt.ylabel('')
        plt.tight_layout()
        if save:
            plt.savefig(name, format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()

    def corr_plots(
        self,
        num_lags: int = 50,
        show: bool = True,
        save: bool = False,
        name: str = 'corr'
    ) -> None:
        """Generate the autocorrelation plots of the posterior samples
        of the parameters of interest. The plots can be displayed in one
        single figure on the screen and/or saved as a picture. A custom
        name/path can be given for the saved file.

        Args:
            self (object): A class instance that has attributes 'full-
                chain', 'nonspat' and '_names' defined.
            num_lags (int, optional): The maximum number of lags to dis-
                play in the plot. Defaults to 50.
            show (bool, optional): Whether to show the plot on screen.
                Defaults to True.
            save (bool, optional): Whether to save plot as a picture.
                Defaults to False.
            name (str, optional): Name of the picture if save = True.
                The supplied input can also be a path if it is to be
                saved in a custom directory. Defaults to 'traces'.
        """
        traces = self.fullchain
        if self.nonspat:
            plot_rows = traces.shape[1] - 1
        else:
            plot_rows = traces.shape[1]
        plt.figure(figsize=(6, 8))
        for i in range(plot_rows):
            data = traces[:, i][1:].astype(np.float64)
            lagdata = [acf(data, lag=i) for i in range(0, num_lags + 1)]
            plt.subplot(
                plot_rows,
                1, i + 1,
                xlim=(-1, num_lags + 1),
                ylim=(min(lagdata) - 0.2 if min(lagdata) < 0 else -0.2, 1.2)
            )
            plt.plot(lagdata, 'C0o', markersize=5)
            ymaxs = [y - 0.05 if y > 0 else y + 0.05 for y in lagdata]
            plt.vlines(np.arange(num_lags + 1), ymin=0, ymax=ymaxs, color='k')
            plt.hlines(y=0, xmin=-1, xmax=num_lags + 1, color='C0')
            plt.title(f"acf of {self._names[i]}")
        plt.tight_layout()
        if save:
            plt.savefig(name, format='svg', bbox_inches='tight')
        plt.show() if show else plt.clf()


class ProgressBar:
    """A class to emulate a progress bar on the terminal

    Args:
        n (int): Total number of iterations.

    Attributes:
        BAR_LENGTH (int): The horizontal length of the progress bar.
        _FILE (io.TextIOWrapper): A file object.
        start (float): The start time, created using time.monotonic()
        n (int): Total number of iterations.
        i (int): The current iteration.
        fill (str): The progress bar fill. Either a solid bar or hash.
        progress (float): A value between 0 and 1 (inclusive) indicating
            the percentage progress.

    Methods:
        update: Updates the progress forward by one iteration.
    """
    BAR_LENGTH = 25

    def __init__(self, n: int) -> None:
        self._FILE = open(sys.stderr.fileno(), mode='w', encoding='utf8')
        self.start = time.monotonic()
        self.n = n
        self.i = 0
        self.fill = "█"
        self.progress = 0

    def _bar_string(self) -> str:
        """Describes the progress bar output string"""
        elapsed, remaining = self._elapsed_and_remaining_time()
        self.progress = self.i / self.n
        block = int(round(ProgressBar.BAR_LENGTH * self.progress))
        bar = [
            self.fill * block + ' ' * (ProgressBar.BAR_LENGTH - block),
            self.progress * 100,
            str(elapsed).split('.')[0],
            self.i,
            self.n,
            str(remaining).split(".")[0],
            round(self.i / elapsed.total_seconds(), 2)
        ]
        return '{1:.1f}%[{0}] {3}/{4} [{2}<{5}, {6}draws/s]'.format(*bar)

    def _elapsed_and_remaining_time(self) -> Tuple[timedelta, timedelta]:
        """Calculates the elapsed time and remaining time"""
        now = time.monotonic()
        elapsed_time = timedelta(seconds=now - self.start)
        est_total_time = timedelta(
            seconds=self.n / self.i * (now - self.start)
        )
        return elapsed_time, est_total_time - elapsed_time

    def update(self) -> None:
        """Update the progress bar output"""
        self.i += 1
        print(self._bar_string(), file=self._FILE, end='\r')
        if self.progress == 1:
            print()
