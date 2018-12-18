# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from multiprocessing import cpu_count

from tensorflow.contrib.tpu import TPUEstimator, TPUEstimatorSpec
from tensorflow.python.data.experimental import parallel_interleave, map_and_batch

import modeling
import optimization
import tensorflow as tf

from eval_results_hook import EvalResultsHook
from export_hook import ExportHook

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "config//bert_config_file.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file",
    "..\\PREnzyme\\data\\protein\\embedding\\sample\\*",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", "weights\\test",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_full_eval", False, "Whether to run eval on the all set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, eval_hook=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(config=bert_config,
                                   is_training=is_training,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   token_type_ids=None,
                                   use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
                                                     use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                          loss=total_loss,
                                                          train_op=train_op,
                                                          scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(labels=masked_lm_ids,
                                                         predictions=masked_lm_predictions,
                                                         weights=masked_lm_weights)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_mean_loss = tf.metrics.mean(values=masked_lm_example_loss,
                                                      weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss
                }

            eval_metrics = (metric_fn, [masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                                        masked_lm_weights])

            n_predictions = masked_lm_ids.get_shape().as_list()[-1]
            probs = tf.reshape(masked_lm_log_probs,
                               [1024, n_predictions, bert_config.vocab_size])
            masked_lm_predictions = tf.argmax(probs, axis=-1, output_type=tf.int32)
            correct_prediction = tf.equal(masked_lm_predictions, masked_lm_ids)
            masked_lm_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=1)
            # tf.summary.scalar("train_accuracy", tf.reduce_mean(masked_lm_accuracy))
            loss_per_seq = tf.reduce_mean(tf.reshape(masked_lm_example_loss, [1024, n_predictions]), axis=1)
            variables_to_export = [input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights,
                                   loss_per_seq, probs, masked_lm_accuracy, features["seq"]]

            output_spec = TPUEstimatorSpec(mode=mode,
                                           loss=total_loss,
                                           #eval_metrics=eval_metrics,
                                           scaffold_fn=scaffold_fn,
                                           evaluation_hooks=[eval_hook(variables_to_export, FLAGS.output_dir)])
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)
    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = tf.identity(numerator / denominator, name="loss")

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     vocab_size,
                     is_training,
                     num_cpu_threads=1):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                parallel_interleave(
                    lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                parallel_interleave(
                    lambda filename: tf.data.TFRecordDataset(filename, compression_type='GZIP'),
                    sloppy=is_training,
                    cycle_length=cycle_length))
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.

            # d = d.repeat()
            # d = d.shuffle(buffer_size=100000)

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            map_and_batch(
                lambda record: _decode_record(record, max_seq_length, max_predictions_per_seq, vocab_size, is_training),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, max_seq_length, max_predictions_per_seq, vocab_size, is_training):
    """Decodes a record to a TensorFlow example."""
    feature = tf.parse_single_example(record, features={
        "length": tf.FixedLenFeature([], tf.int64),
        'seq': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    })

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(feature.keys()):
        t = feature[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        feature[name] = t

    feature["input_mask"] = pad_up_to(tf.ones(feature["length"]), [max_seq_length], dynamic_padding=False)
    feature["input_mask"].set_shape([max_seq_length])
    feature["seq"] = pad_up_to(feature["seq"], [max_seq_length], dynamic_padding=False)
    feature["seq"].set_shape([max_seq_length])
    feature["input_ids"] = feature["seq"]

    if is_training:

        positions_to_mask = tf.cond(tf.greater(tf.random.uniform(minval=0, maxval=1, shape=[]), 0.95),
                                    lambda: tf.random.uniform([max_predictions_per_seq], 0, [max_seq_length]),
                                    lambda: tf.random.uniform([max_predictions_per_seq], 0, [feature["length"]]))
        positions_to_mask = tf.cast(positions_to_mask, tf.int32)
    else:
        positions_to_mask = tf.cast(tf.random.uniform([max_predictions_per_seq], 0, [feature["length"]]), tf.int32)

    feature["masked_lm_positions"] = positions_to_mask
    feature["masked_lm_ids"] = tf.gather(feature["input_ids"], positions_to_mask)
    feature["masked_lm_weights"] = tf.ones(max_predictions_per_seq, dtype=tf.float32)

    if is_training:
        r = tf.random.uniform(minval=-28, maxval=170, shape=[max_predictions_per_seq], dtype=tf.int32)
        added = tf.add(feature["masked_lm_ids"], r)
        negative_mask = tf.less(added, tf.constant(0))
        to_subtract = tf.multiply(r, tf.cast(negative_mask, tf.int32))
        mask = tf.subtract(r, to_subtract)
    else:
        mask = tf.ones([max_predictions_per_seq], dtype=tf.int32) * vocab_size

    to_mask = tf.scatter_nd(tf.expand_dims(positions_to_mask, axis=1), mask, [max_seq_length])
    feature["input_ids"] = tf.clip_by_value(tf.add(feature["input_ids"], to_mask), 0, 21)

    # feature.pop("seq", None)
    return feature


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_full_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=FLAGS.iterations_per_loop,
                                            num_shards=FLAGS.num_tpu_cores,
                                            per_host_input_for_training=is_per_host))
    eval_hook = None
    if FLAGS.do_eval:
        eval_hook = ExportHook
    if FLAGS.do_full_eval:
        eval_hook = EvalResultsHook
    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=FLAGS.num_train_steps,
                                num_warmup_steps=FLAGS.num_warmup_steps,
                                use_tpu=FLAGS.use_tpu,
                                use_one_hot_embeddings=FLAGS.use_tpu,
                                eval_hook=eval_hook)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.

    estimator = TPUEstimator(use_tpu=FLAGS.use_tpu,
                             model_fn=model_fn,
                             config=run_config,
                             train_batch_size=FLAGS.train_batch_size,
                             eval_batch_size=FLAGS.eval_batch_size)

    input_fn = input_fn_builder(input_files=input_files,
                                max_seq_length=FLAGS.max_seq_length,
                                max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                vocab_size=bert_config.vocab_size,
                                is_training=FLAGS.do_train,
                                num_cpu_threads=cpu_count() - 1)
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        estimator.train(input_fn=input_fn, max_steps=FLAGS.num_train_steps,
                        # hooks=[tf.train.LoggingTensorHook(tensors={'loss': 'cls/predictions/loss'}, every_n_iter=1)]
                        )
    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        result = estimator.evaluate(
            input_fn=input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_full_eval:
        tf.logging.info("***** Running full evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        result = estimator.evaluate(input_fn=input_fn, steps=FLAGS.max_eval_steps)


def calculate(current_shape, target_shape, dynamic_padding=False):
    if dynamic_padding:
        missing_padding = target_shape - current_shape

        def empty_padding():
            return [missing_padding, missing_padding]

        def random_padding():
            front = tf.squeeze(
                tf.random_uniform([1], minval=0, maxval=missing_padding, name="random_padding", dtype=tf.int32))
            end = missing_padding - front
            return [front, end]

        return tf.cond(tf.equal(missing_padding, tf.constant(0)),
                       empty_padding,
                       random_padding)

    else:
        return [0, target_shape - current_shape]


def pad_up_to(x, output_shape, constant_values=0, dynamic_padding=False):
    """

    Args:
      x: Input tensor
      output_shape: Output shape
      constant_values:  Values used for padding (Default value = 0)

    Returns:
        Returns padded tensor that is the shape of output_shape.
    """
    s = tf.shape(x)
    paddings = [calculate(s[i], m, dynamic_padding) for (i, m) in enumerate(output_shape)]
    return tf.pad(x, paddings, 'CONSTANT', constant_values=constant_values)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
