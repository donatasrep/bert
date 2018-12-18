import threading

import numpy
from datetime import datetime
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
import os

class EvalResultsHook(SessionRunHook):

    def __init__(self, args_to_store, output_dir):
        self.args_to_store = [ args_to_store[i] for i in [5,6,7,8]]
        self.output_dir = output_dir
        os.makedirs(os.path.join("weights", "full_eval"), exist_ok=True)


    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self.args_to_store)

    def after_run(self, run_context, run_values):
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = os.path.join("weights" "full_eval", filename+".npz")
        values = run_values.results
        Storage(filename, values).run()


class Storage(threading.Thread):
    def __init__(self, filename, values):
        self.filename = filename
        self.values = values
        threading.Thread.__init__(self)

    def run(self):
        try:
            with open(self.filename, 'wb') as f:
                amino_acid_preds = self.values[1]
                sorted = numpy.sort(amino_acid_preds, axis=2)
                score = sorted[:, :, -1] - sorted[:, :, 3]
                score = numpy.mean(score, axis=1)
                self.values[1] = score
                numpy.savez(f, loss=self.values[0], score=self.values[1], acc=self.values[2], seq=self.values[3])
            print("Finished saving {} file".format(self.filename))
        except Exception as e:
            print("Unexpected error while saving to numpy file:", str(e))