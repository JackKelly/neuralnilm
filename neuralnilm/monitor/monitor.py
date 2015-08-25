from __future__ import print_function, division
from time import sleep
import pymongo
from monary import Monary
import numpy as np
import matplotlib.pyplot as plt
from neuralnilm.utils import downsample


class Monitor(object):
    def __init__(self, experiment_id, update_period=1, max_lines=1000):
        """
        Parameters
        ----------
        max_lines : int
            Number of pixels.
        """
        self.experiment_id = experiment_id
        self.update_period = update_period
        self.max_lines = max_lines
        self._last_training_iteration_processed = 0
        self.mongo_client = pymongo.MongoClient()
        self.db = self.mongo_client.neuralnilm_experiments

    def start(self):
        while True:
            if self._new_training_costs_available():
                self._plot_training_costs()
            sleep(self.update_period)

    def _new_training_costs_available(self):
        """Returns True is new training costs are available from DB."""
        row = self.db.train_scores.find_one(
            filter={
                'experiment_id': self.experiment_id,
                'iteration': {'$gt': self._last_training_iteration_processed}
            }
        )
        return (row is not None)

    def _plot_training_costs(self):
        monary = Monary()
        iterations, loss = monary.query(
            db='neuralnilm_experiments',
            coll='train_scores',
            query={
                'experiment_id': self.experiment_id
            },
            fields=['iteration', 'loss'],
            types=['int32', 'float32']
        )

        # Downsample if necessary
        if loss.size > self.max_lines:
            divisor = int(np.ceil(loss.size / self.max_lines))
            loss = downsample(loss, divisor)
            iterations = np.arange(iterations[0], iterations[-1], divisor)

        fig, ax = plt.subplots(1)
        ax.plot(iterations, loss)
        plt.show()
        self._last_training_iteration_processed = iterations[-1]
