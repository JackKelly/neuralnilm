from __future__ import print_function, division


class BatchSeq(object):
    def __init__(self):
        self.input = None
        self.target = None


class Batch(object):
    """
    Attributes
    ----------
    all_appliances : pd.DataFrame
        2-level hierarchical column names: seq_i, appliance_name
    metadata : dict
    before_processing : BatchSeq
    after_processing : BatchSeq
    weights : None or np.ndarray
    """
    def __init__(self):
        self.all_appliances = []
        self.metadata = {}
        self.before_processing = BatchSeq()
        self.after_processing = BatchSeq()
        self.weights = None

    @property
    def input(self):
        return self.after_processing.input

    @property
    def target(self):
        return self.after_processing.target
