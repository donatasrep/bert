import numpy
import os
from datetime import datetime
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs


class EvalResultsHook(SessionRunHook):

    def __init__(self, args_to_store, output_dir):
        self.args_to_store = list([args_to_store[i] for i in [5,7,8]])
        filename = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(output_dir, "eval_full"), exist_ok=True)
        self.filename = os.path.join(output_dir, "eval_full", filename+".npy")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self.args_to_store)

    def after_run(self, run_context, run_values):
        values = run_values.results
        with open(self.filename, 'wb') as f:
            numpy.save(f, values)