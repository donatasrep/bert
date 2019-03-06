# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Batching dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_libs.data.experimental.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tf_libs.data.experimental.ops import gen_dataset_ops


class _MapAndBatchDataset(dataset_ops.MapDataset):
  """A `Dataset` that maps a function over a batch of elements."""

  def __init__(self, input_dataset, map_func, batch_size, num_parallel_calls,
               drop_remainder):
    """See `Dataset.map()` for details."""
    super(_MapAndBatchDataset, self).__init__(input_dataset, map_func)
    self._batch_size_t = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._num_parallel_calls_t = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    self._drop_remainder_t = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    self._batch_size = batch_size
    self._drop_remainder = drop_remainder

  def _as_variant_tensor(self):
    # pylint: disable=protected-access
    input_resource = self._input_dataset._as_variant_tensor()
    return gen_dataset_ops.map_and_batch_dataset_v2(
        input_resource,
        self._map_func.captured_inputs,
        f=self._map_func,
        batch_size=self._batch_size_t,
        num_parallel_calls=self._num_parallel_calls_t,
        drop_remainder=self._drop_remainder_t,
        **dataset_ops.flat_structure(self))
    # pylint: enable=protected-access

  @property
  def output_shapes(self):
    dim = self._batch_size if self._drop_remainder else None
    return nest.pack_sequence_as(self._output_shapes, [
        tensor_shape.vector(dim).concatenate(s)
        for s in nest.flatten(self._output_shapes)
    ])

  @property
  def output_types(self):
    return self._output_types


def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None):
  """Fused implementation of `map` and `batch`.

  Maps `map_func` across `batch_size` consecutive elements of this dataset
  and then combines them into a batch. Functionally, it is equivalent to `map`
  followed by `batch`. However, by fusing the two transformations together, the
  implementation can be more efficient. Surfacing this transformation in the API
  is temporary. Once automatic input pipeline optimization is implemented,
  the fusing of `map` and `batch` will happen automatically and this API will be
  deprecated.

  Args:
    map_func: A function mapping a nested structure of tensors to another
      nested structure of tensors.
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
      representing the number of batches to create in parallel. On one hand,
      higher values can help mitigate the effect of stragglers. On the other
      hand, higher values can increase contention if CPU is scarce.
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in case its size is smaller than
      desired; the default behavior is not to drop the smaller batch.
    num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number of elements to process in parallel. If not
        specified, `batch_size * num_parallel_batches` elements will be
        processed in parallel.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
      specified.
  """

  if num_parallel_batches is None and num_parallel_calls is None:
    num_parallel_calls = batch_size
  elif num_parallel_batches is not None and num_parallel_calls is None:
    num_parallel_calls = batch_size * num_parallel_batches
  elif num_parallel_batches is not None and num_parallel_calls is not None:
    raise ValueError("The `num_parallel_batches` and `num_parallel_calls` "
                     "arguments are mutually exclusive.")

  def _apply_fn(dataset):
    return _MapAndBatchDataset(dataset, map_func, batch_size,
                               num_parallel_calls, drop_remainder)

  return _apply_fn
