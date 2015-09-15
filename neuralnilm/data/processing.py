from __future__ import division
from copy import copy


class Processor(object):
    def report(self):
        report = copy(self.__dict__)
        report['name'] = self.__class__.__name__
        return report

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


class IndependentlyCenter(Processor):
    def __call__(self, data):
        means = data.mean(axis=1, keepdims=True)
        self.metadata = {'IndependentlyCentre': {'means': means}}
        return data - means
