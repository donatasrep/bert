import numpy
from datetime import datetime
import os
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs


class ExportHook(SessionRunHook):

    def __init__(self, args_to_store, output_dir):
        # input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights,
        # loss_per_seq, probs, masked_lm_accuracy, features["seq"]
        self.args_to_store = [args_to_store[i] for i in [5, 6, 7, 8, 3]]
        filename = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(output_dir, "eval_results"), exist_ok=True)
        self.filename = os.path.join(output_dir, "eval_results", filename + ".npz")
        self.values = None

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self.args_to_store)

    def after_run(self, run_context, run_values):
        if self.values is None:
            self.values = run_values.results
        else:
            values = run_values.results
            for i in range(len(self.values)):
                self.values[i] = numpy.concatenate((self.values[i], values[i]))

    def end(self, session):
        try:
            with open(self.filename, 'wb') as f:
                numpy.savez(f, loss=self.values[0], probs=self.values[1], acc=self.values[2], seq=self.values[3], correct=self.values[4])
            print("Finished saving {} file".format(self.filename))
        except Exception as e:
            print("Unexpected error while saving to numpy file:", str(e))
