# TODO
# input ellenorzese
# test cases
# pylint proba
# mini pelda
# valtozotipusok?
# parameterezheto input-output fajtak *int128-tol int 16-ig (!!! es elojeles,
#   nem elojeles input-output)
# input: int32 jo, uint NEM, tobbi int NEM: tudni kell, mikor elerheto a
#   bemenet tipusa
# output: float16, bfloat16 igen, tobbit tesztelni kell
# shape parameter, amivel a (...y, 4) - bol (...4y) lesz es viszont
#    (inputkent is fogadni visszakodolaskor)
# scaling algoritmus, visszafele is mukodjon,
#   illetve opcionalis -1 1 (tanh), 0-1 hatarokkal (sigmoid)
# eager / non eager teszt

# Tamas Burghard
# © 2023. This work is licensed under the `Unlicense`license.
#
"""Int-float Tensorflow layers that could keep all bits of information.


  The rationale behind is to keep all of the bit information of potentially
  very big integer inputs for a deep learning model. This could be achieved by
  simply converting the numbers to Float64 and rescaling, however recent
  architectures tend to use less precision.
  Converting an `Int32` to 32 floats bit-by-bit seems to be too many.
  This solution is to convert an `Int32` to 4 floats.

  These layers take the input integers(for instance an Int32), and byte-by-byte
  convert them to floats.
  These layers intended to being used only in experimentation.

  A bytestream input is preferred over this implementation in production.

  Note: Dividing by 127.5 (or 255) is numerically less stable than dividing by 128. 
  The float->int backward conversation could use an optional `sigmoid` or `tanh`  
  and a multiplication by 255 . This difference makes no
  problem, as the two layers are not meant to be numerical inverses.
  The point here is the possibility to keep all the information from the
  inputs and the possibility to assemble similar -potentially big- numbers at
  the end of the pipeline.

  Using these layers make sense if the exact representation is needed through
  the process. In these cases, most probably the task depends on the order of
  the numbers, so the usual loss functions are not well situated.
  The guarantee here is:
    - You can exactly represent all 32bit integers in the input phase.
    - The same for the output phase. This does NOT mean that those
      representations have to be the same.

  Note: As the inputs and the outputs are

  Warning: tf.bitcast is being used.
  Warning: intended to use with simple tensors.
  Warning: the type `bfloat16` is not supported with shift_value != 0.0
"""

# bfloat16 eseten 127-el osztas eseten csak a 255-lesz rosszul prezentalva,
# ezt egy clip megoldja, mert
# 256 lesz az oda-vissza konverzio utan.

# megoldas: tf.saturate_cast

from typing import Optional

import tensorflow as tf

logger = tf.get_logger()


class Int32ToFloatLayer(tf.keras.layers.Layer):
    """Converts every byte of `int32` tensors to floats (optional scaling).

    The `Int32ToFloatLayer` class is a subclass of `tf.keras.layers.Layer`, so it
    can be used as a layer in a Keras model. The layer takes an `int32` tensor as
    input, and returns a `float32` (or the specified type) tensor as output.

    There is a default scaling algorithm, that is dividing the input by 127.5, and
    using 1.0 as a shift value. (0-1 limits) There are nummerically more stable
    constants for the scaling, but the default values are more intuitive.


    Args:
        reshape: A boolean indicating whether to reshape the output tensor.
            If `True`, the last dimension of the output tensor is multiplied by 4.
            Default is `False`, meaning that the output tensor has a new dimension
            appended to the end. (size 4)
        target_dtype: A `tf.DType` object indicating the target data type for the
            output tensor. The possible values are `tf.float16`, `tf.float32`,
            `tf.float64`, and `tf.bfloat16`. Default is `tf.float32`.
        verbose: A boolean indicating whether to print debug information during
            the forward pass. Default is `True`.
        scaler_value: A float value indicating the scaling factor for the
            conversion. We divide with this number. Default is `127.5`.
        shift_value: A float value indicating the shift value for the conversion.
            We substract this number here. Default is `1.0`.
        **kwargs: Keyword arguments to be passed to `tf.keras.layers.Layer`.

    Raises:
        AssertionError: If the `target_dtype` argument is not one of the tested
            data types.

    """

    TESTED_DTYPES = (tf.float16, tf.float32, tf.float64, tf.bfloat16)

    def __init__(
        self,
        reshape: bool = False,
        target_dtype: tf.DType = tf.float32,
        verbose: bool = True,
        scaler_value: float = 127.5,
        shift_value: float = 1.0,
        **kwargs,
    ):
        super(Int32ToFloatLayer, self).__init__(**kwargs)
        self.reshape = reshape
        assert target_dtype in self.TESTED_DTYPES
        self._target_dtype = target_dtype  # TODO: mixed precision
        self.scaler_value = tf.constant(scaler_value, target_dtype)
        self.shift_value = tf.constant(shift_value, target_dtype)
        self.verbose = verbose
        if verbose:
            print(f"int32 init dtype:{self.dtype}")

    def call(self, x):
        if self.verbose:
            print(f"x type {x.dtype}")
            print(f"x shape IN {x.shape}")
        assert x.dtype == tf.int32
        x_bytes = tf.bitcast(x, tf.uint8)
        if self.verbose:
            print(f"converted to bytes:  {x_bytes}")
        x_float = (
            tf.saturate_cast(x_bytes, self._target_dtype) / self.scaler_value
            - self.shift_value
        )
        if self.verbose:
            print(f"x shape OUT {x_float.shape}")
        if self.reshape:
            shp = x_float.shape
            x_float = tf.reshape(x_float, shp[:-2] + [shp[-2] * 4])
        return x_float

    def compute_output_shape(self, input_shape):
        if self.reshape:
            shp = input_shape[:-1] + [4 * input_shape[-1]]
        else:
            shp = input_shape + [4]
        return shp


class FloatToInt32Layer(tf.keras.layers.Layer):
    """
    A custom Keras layer that converts float32 tensors to int32 tensors.

    The main concern of this layer is to scale the input tensor to the range
    [0, 255] and then convert it to `int32`.
    By default, no nonlinearity is applied to the input tensor, and this layer
    is the exact inverse of the `Int32ToFloatLayer` layer. However, if you
    want to use this layer as a part of a neural network, you might want to apply
    a nonlinearity function to the input tensor, `tf.sigmoid` for example.

    The order of the operations is:
    1. Apply the nonlinearity function (optional)
    2. Scale the input tensor to the range [0, 255]
    3. Convert the input tensor to `int32` from 4 numbers (bytes).

    Args:
        reshape: Whether to reshape the output tensor. Def: `False`.
        nonlinearity_fn (callable, optional): A nonlinearity function to apply to the
            input tensor. Def: `None`: no nonlinearity is applied -
            you might choose this to get your original input back. You still have to
            specify the scaling and shifting parameters.
        scaler_value: The scaling factor for the conversion. Def: `127.5`. We multiply
            the input tensor with this value, after applying the shift value.
        shift_value: The shift value for the conversion. Def: `0.0`. We add this value
            to the input tensor after applying the nonlinearity function.
        verbose: Whether to print debug information. Def: `False`.
    """

    def __init__(
        self,
        reshape: bool = False,
        nonlinearity_fn: Optional[callable] = None,
        verbose: bool = False,
        scaler_value: float = 127.5,
        shift_value: float = 0.0,
        **kwargs,
    ):
        super(FloatToInt32Layer, self).__init__(**kwargs)
        self.reshape = reshape
        self.nonlinearity_fn = nonlinearity_fn
        self.verbose = verbose
        self.scaler_value = tf.constant(127.5, tf.float32)
        self.shift_value = tf.constant(shift_value, tf.float32)

    def call(self, x):
        # converting the constants to the actual dtype of the input
        # the dtype is not available in the init method
        self.scaler_value = tf.cast(self.scaler_value, x.dtype)
        self.shift_value = tf.cast(self.shift_value, x.dtype)

        if self.verbose:
            print(f"deconv input {x}")
        x_out = x
        if self.nonlinearity_fn:
            x_out = self.nonlinearity_fn(x)
        if self.reshape:
            new_shape = x_out.shape
            new_shape = new_shape[:-1] + [new_shape[-1] // 4] + [4]
            if self.verbose:
                print(f"reshaping from {x_out.shape} to {new_shape}")
            x_out = tf.reshape(x_out, new_shape)
        # Convert float to bytes
        if self.verbose:
            print(f"after rescaling {(x_out + self.shift_value )* self.scaler_value}")
        # round is not working well for bfloat16 and scaling
        x_bytes = tf.cast(
            tf.round((x_out + self.shift_value) * self.scaler_value), tf.uint8
        )
        if self.verbose:
            print(x_bytes)
        # Convert bytes to int32
        x_int32 = tf.bitcast(x_bytes, tf.int32)
        if self.verbose:
            print(f"deconv output {x_int32}")
        return x_int32

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 4)
