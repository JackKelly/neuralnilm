class Metrics(object):
    def __init__(self, state_boundaries):
        """
        Parameters
        ----------
        state_boundaries : list of numbers
            Define the state (power) boundaries.  For a two-state (on/off)
            appliance, this will be a list of [<on_power_threshold>].
        """
        self.state_boundaries = state_boundaries

    def compute_metrics(self, output, target):
        """
        Returns
        -------
        scores : dict
            {'regression_metrics': {'mse': <>},
             '2_state_classification_metrics': {'f1': <>, ...},
             '3_state_classification_metrics': {}
            }
        """
        scores = {}
        return scores
