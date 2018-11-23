import numpy
import pickle
from datetime import datetime
import os
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs


class ExportHook(SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self, args_to_store, output_dir):
        self.args_to_store = args_to_store
        filename = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs( os.path.join(output_dir, "eval_results"), exist_ok=True)
        self.filename = os.path.join(output_dir, "eval_results", filename+".npy")


    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self.args_to_store)

    def after_run(self, run_context, run_values):
        values = run_values.results
        with open(self.filename, 'ab+') as f:
            numpy.save(f, values)