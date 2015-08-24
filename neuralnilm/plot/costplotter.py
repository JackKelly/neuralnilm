from __future__ import print_function, division
from multiprocessing import Process, Event
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


class CostPlotter(Process):
    def __init__(self, database, experiment_id):
        super(CostPlotter, self).__init__()
        self.database = database
        self.experiment_id = experiment_id
        self._stop_event = Event()
        self._plot_event = Event()

    def run(self):
        logger.info("Starting CostPlotter.")
        try:
            while True:
                self._plot_event.wait()
                if self._stop_event.is_set():
                    break
                self._draw_plot()
                self._plot_event.clear()
        except:
            logger.exception("")
        finally:
            logger.info("Stopping CostPlotter.")

    def _draw_plot(self):
        training_costs = self.database.train_scores.find(
            filter={'experiment_id': self.experiment_id},
            sort=[('iteration', pymongo.ASCENDING)],
            projection=['iteration', 'loss']
        )
        try:
            n_iterations = len(training_costs)
        except:
            return
        validation_costs = self._load_csv('validation_costs.csv')
        validation_interval = int(round(n_iterations / len(validation_costs)))
        SIZE = 2
        fig, ax = plt.subplots(1)

        # Plot training costs
        train_x = np.arange(0, n_iterations)
        ax.scatter(train_x, training_costs, label='Training',
                   alpha=0.2, s=SIZE, linewidths=0)

        # Plot validation costs
        validation_x = np.arange(0, n_iterations, validation_interval)
        n_validations = min(len(validation_x), len(validation_costs))
        ax.scatter(validation_x[:n_validations],
                   validation_costs[:n_validations],
                   label='Validation', c='g', s=SIZE, linewidths=0)

        # Text and formatting
        ax.set_xlim((0, n_iterations))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.grid(True)

        # Save
        plt.savefig(join(self.path, 'costs.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

    def _load_csv(self, filename):
        return np.loadtxt(join(self.path, filename))

    def plot(self):
        self._plot_event.set()

    def stop(self):
        self._stop_event.set()
        self._plot_event.set()
        self.terminate()
        self.join()
