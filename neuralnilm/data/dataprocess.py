from multiprocessing import Process, Queue, Event
from Queue import Empty


class DataProcess(Process):
    def __init__(self, data_pipeline):
        super(DataProcess, self).__init__(name='neuralnilm-data-process')
        self._stop = Event()
        self._queue = Queue(maxsize=3)
        self.data_pipeline = data_pipeline

    def run(self):
        while not self._stop.is_set():
            batch = self.data_pipeline.get_batch()
            self._queue.put(batch)

    def get_batch(self, timeout=30):
        if self.is_alive():
            return self._queue.get(timeout=timeout)
        else:
            raise RuntimeError("Process is not running!")

    def stop(self):
        self._empty_queue()
        self._stop.set()
        self.join()

    def _empty_queue(self):
        while True:
            try:
                self._queue.get(block=False)
            except Empty:
                break
