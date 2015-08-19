from __future__ import division


class Processor(object):
    def report(self):
        return {self.__class__.__name__: self.__dict__}

    def inverse(self, data):
        raise NotImplementedError("To be implemented by subclass.")

    def __call__(self, data):
        raise NotImplementedError("To be implemented by subclass.")


class DivideBy(Processor):
    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, dividend):
        return dividend / self.divisor

    def inverse(self, quotient):
        return quotient * self.divisor
