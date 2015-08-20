from __future__ import print_function
from multiprocessing import Process, Queue, Event


class DataProcess(Process):
    def __init__(self, data_pipeline, **get_batch_kwargs):
        super(DataProcess, self).__init__(name='neuralnilm-data-process')
        self._stop = Event()
        self._queue = Queue(maxsize=3)
        self.data_pipeline = data_pipeline
        self._get_batch_kwargs = get_batch_kwargs

    def run(self):
        batch = self.data_pipeline.get_batch(**self._get_batch_kwargs)
        while not self._stop.is_set():
            try:
                self._queue.put(batch)
            except AssertionError:
                # queue is closed
                break
            batch = self.data_pipeline.get_batch(**self._get_batch_kwargs)

    def get_batch(self, timeout=30):
        if self.is_alive():
            return self._queue.get(timeout=timeout)
        else:
            raise RuntimeError("Process is not running!")

    def stop(self):
        self._stop.set()
        self._queue.close()
        self.terminate()
        self.join()
