class Pipeline(object):
    def __init__(self, source):
        self.source = source
        self.pipeline = []

    def run(self, validation=False):
        data = self.source.get_data(validation=validation)
        for processor in self.pipeline:
            data = processor(data)
