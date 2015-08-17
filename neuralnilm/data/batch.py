from __future__ import print_function, division


class BatchSeq(object):
    def __init__(self):
        self.input = None
        self.target = None


class Batch(object):
    """
    Attributes
    ----------
    all_appliances : list of pd.DataFrames
        One entry for each seq in batch
    metadata : dict
    before_processing : BatchSeq
    after_processing : BatchSeq
    """
    def __init__(self):
        self.all_appliances = []
        self.metadata = {}
        self.before_processing = BatchSeq()
        self.after_processing = BatchSeq()
