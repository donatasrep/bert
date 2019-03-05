"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: dataset_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops.gen_dataset_ops import _op_def_lib

def map_and_batch_dataset_v2(input_dataset, other_arguments, batch_size, num_parallel_calls, drop_remainder, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that fuses mapping with batching.

  Creates a dataset that applies `f` to the outputs of `input_dataset` and then

  batches `batch_size` of them.

  

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up

  to `batch_size * num_parallel_batches` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when building a closure

      for `f`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a

      batch. It determines the number of concurrent invocations of `f` that process

      elements from `input_dataset` in parallel.
    num_parallel_calls: A `Tensor` of type `int64`.
      A scalar representing the maximum number of parallel invocations of the `map_fn`

      function. Applying the `map_fn` on consecutive input elements in parallel has

      the potential to improve input pipeline throughput.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size

      is smaller than desired.
    f: A function decorated with @Defun.
      A function to apply to the outputs of `input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(output_types, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_types' argument to "
          "'map_and_batch_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_shapes' argument to "
          "'map_and_batch_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
    _, _, _op = _op_def_lib._apply_op_helper(
        "MapAndBatchDatasetV2", input_dataset=input_dataset,
        other_arguments=other_arguments, batch_size=batch_size,
        num_parallel_calls=num_parallel_calls, drop_remainder=drop_remainder,
        f=f, output_types=output_types, output_shapes=output_shapes,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _execute.record_gradient(
      "MapAndBatchDatasetV2", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "MapAndBatchDatasetV2", name, _ctx._post_execution_callbacks,
        input_dataset, other_arguments, batch_size, num_parallel_calls,
        drop_remainder, "f", f, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._FallbackException:
      return map_and_batch_dataset_v2_eager_fallback(
          input_dataset, other_arguments, batch_size, num_parallel_calls,
          drop_remainder, f=f, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def map_and_batch_dataset_v2_eager_fallback(input_dataset, other_arguments, batch_size, num_parallel_calls, drop_remainder, f, output_types, output_shapes, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function map_and_batch_dataset_v2
  """
  _ctx = ctx if ctx else _context.context()
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_and_batch_dataset_v2' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_and_batch_dataset_v2' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
  num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
  drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
  _inputs_flat = [input_dataset] + list(other_arguments) + [batch_size, num_parallel_calls, drop_remainder]
  _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
  output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"MapAndBatchDatasetV2", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "MapAndBatchDatasetV2", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


