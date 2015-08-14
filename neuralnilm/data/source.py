from __future__ import print_function, division


class Seq(object):
    """
    Attributes
    ----------
    input : np.ndarray
    target : np.ndarray
    all_appliances : pd.DataFrame
        columns names are the appliance names
    """
    def __init__(self):
        self.input = None
        self.target = None
        self.all_appliances = None


class Source(object):
    def get_seq(self):
        """
        Returns
        -------
        seq : Seq
        """
        raise NotImplementedError()
