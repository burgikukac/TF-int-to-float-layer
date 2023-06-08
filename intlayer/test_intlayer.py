"""Tests for the intlayer module."""
import itertools
import sys

import numpy as np
import pytest
import tensorflow as tf

from .intlayer import FloatToInt32Layer, Int32ToFloatLayer

if "ipykernel" in sys.modules:
    # Running in a Jupyter notebook
    from tqdm.notebook import tqdm
else:
    # Not running in a Jupyter notebook
    from tqdm import tqdm


def run_test_int32_to_floats(P, chunk_size=2**24):
    pbar = tqdm(total=2**25)
    int32_to_float16_layer = Int32ToFloatLayer(verbose=False, **P)
    float16_to_int32_layer = FloatToInt32Layer(verbose=False, **P, nonlinearity_fn=None)

    # for i in range(tf.int32.min, tf.int32.max, chunk_size):

    for i in range(tf.int32.min, tf.int32.min + chunk_size, chunk_size):
        x_chunk = tf.range(
            start=i, limit=min(i + chunk_size, tf.int32.max), dtype=tf.int32
        )
        f_chunk1 = int32_to_float16_layer(x_chunk)
        x_chunk2 = float16_to_int32_layer(f_chunk1)
        assert np.all(x_chunk == x_chunk2), (
            f"Conversion failed for chunk starting at x = {i}\n"
            f"First failed number: {x_chunk[np.argmax(x_chunk != x_chunk2)]}\n"
            f"Failed with params: {P}"
        )
        pbar.update(chunk_size)
    pbar.close()
    print("All conversions successful")


def _test_long_test():
    # this test is intended for running by hand
    param_dict = {
        "reshape": [True, False],
        "dtype": Int32ToFloatLayer.TESTED_DTYPES[:-1],
        "dims": [1, 2, 3, 4, 5],
    }
    param_generator = (
        dict(zip(param_dict, x)) for x in itertools.product(*param_dict.values())
    )

    for P in list(param_generator):
        print(P)
        P.pop("dims")
        run_test_int32_to_floats(P)


@pytest.mark.parametrize(
    "reshape, target_dtype, test_size, chunk_size, shift_value",
    itertools.product(
        [True, False],
        Int32ToFloatLayer.TESTED_DTYPES,
        [2**24 - 1],
        [2**24],
        [0.0, 1.0],
    ),
)
def test_int32_to_floats(reshape, target_dtype, test_size, chunk_size, shift_value):
    if shift_value == 1.0 and target_dtype == tf.bfloat16:
        pytest.xfail("This combination is expected to fail")
    int32_to_float16_layer = Int32ToFloatLayer(
        reshape=reshape, dtype=target_dtype, verbose=False, shift_value=shift_value
    )
    float16_to_int32_layer = FloatToInt32Layer(
        reshape=reshape,
        dtype=target_dtype,
        verbose=False,
        nonlinearity_fn=None,
        shift_value=shift_value,
    )

    for i in range(tf.int32.min, tf.int32.min + test_size, chunk_size):
        x_chunk = tf.range(
            start=i, limit=min(i + chunk_size, tf.int32.max), dtype=tf.int32
        )
        f_chunk1 = int32_to_float16_layer(x_chunk)
        x_chunk2 = float16_to_int32_layer(f_chunk1)
        assert np.all(x_chunk == x_chunk2), (
            f"Conversion failed for chunk starting at x = {i}\n"
            f"First failed number: {x_chunk[np.argmax(x_chunk != x_chunk2)]}\n"
            f"Failed with params: {reshape}, {target_dtype}, {test_size}, {chunk_size},"
            f" {shift_value}"
        )
    print("All conversions were successful")


def test_int32_to_float_layer_no_reshape():
    layer = Int32ToFloatLayer(reshape=False)
    x = tf.ones((10,), dtype=tf.int32)
    y = layer(x)
    assert y.shape == (10, 4), f"Expected output shape (10, 4), but got {y.shape}"


def test_int32_to_float_layer_with_reshape():
    layer = Int32ToFloatLayer(reshape=True)
    x = tf.ones((10,), dtype=tf.int32)
    y = layer(x)
    assert y.shape == (40,), f"Expected output shape (40, ), but got {y.shape}"


def test_float_to_int32_layer_no_reshape():
    layer = FloatToInt32Layer(reshape=False)
    x = tf.ones((10, 4), dtype=tf.float32)
    y = layer(x)
    assert y.shape == (10,), f"Expected output shape (10,), but got {y.shape}"


def test_float_to_int32_layer_with_reshape():
    layer = FloatToInt32Layer(reshape=True)
    x = tf.ones((40,), dtype=tf.float32)
    y = layer(x)
    assert y.shape == (10,), f"Expected output shape (10,), but got {y.shape}"


def test_always_pass():
    pass
