# TODO
# input ellenorzese
# test cases
# pylint proba
# mini pelda
# valtozotipusok?
# parameterezheto input-output fajtak *int128-tol int 16-ig (!!! es elojeles, nem elojeles input-output)
# input: int32 jo, uint NEM, tobbi int NEM: tudni kell, mikor elerheto a bemenet tipusa
# output: float16, bfloat16 igen, tobbit tesztelni kell          
# shape parameter, amivel a (...y, 4) - bol (...4y) lesz es viszont (inputkent is fogadni visszakodolaskor)
# scaling algoritmus, visszafele is mukodjon, illetve opcionalis -1 1 (tanh), 0-1 hatarokkal (sigmoid)
# eager / non eager teszt

# Tamas Burghard
# © 2023. This work is licensed under a CC BY 4.0 license. 
#
"""Int-float Tensorflow layers that could keep all bits of information.


  The rationale behind is to keep all of the bit information of potentially very
  big integer inputs for a deep learning model. This could be achieved by simply
  converting the numbers to Float64 and rescaling, however recent architectures 
  tend to use less precision. 
  These layers take the input integers(for instance an Int32), and byte-by-byte
  convert them to floats. 
  These layers intended to being used only in experimentation.

  A bytestream input is preferred over this implementation in production.

  Note: the scaling part of the forward and backward conversion are a bit
  different. A vanilla scaler would divide by 127 (or 255), that is numerically
  less stable than dividing by 128. The float->int backward conversation uses
  an optional tanh and a multiplication by 255. This difference makes no
  problem, as the two layers are not meant to be numerical inverses.
  The point here is the possibility to keep all the information from the inputs, 
  and the possibility to assemble similar -potentially big- numbers at the end
  of the pipeline. 

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
"""

# bfloat16 eseten 127-el osztas eseten csak a 255-lesz rosszul prezentalva, ezt egy clip megoldja, mert
# 256 lesz az oda-vissza konverzio utan. 

# megoldas: tf.saturate_cast

import tensorflow as tf
logger = tf.get_logger()

class Int32ToFloatLayer(tf.keras.layers.Layer):
    """Transforms every byte of `int32` tensors to floats with optional scaling."""

    TESTED_DTYPES = (tf.float16, tf.float32, tf.float64, tf.bfloat16)

    def __init__(self, reshape: bool = False, target_dtype: tf.DType = tf.float32, verbose=True, **kwargs):
        super(Int32ToFloatLayer, self).__init__(**kwargs)

        self.reshape = reshape
        assert target_dtype in self.TESTED_DTYPES # rename if class name is fixed
        self._target_dtype = target_dtype # TODO: mixed precision
        self._CONST_127 = tf.constant(127.5, target_dtype)
        self._ONE = tf.constant(1.0, target_dtype)
        
        self.verbose = verbose
        if verbose: print(f'int32 init dtype:{self.dtype}')

    def call(self, x):
        if self.verbose:
          print(f'x type {x.dtype}')
          print(f'x shape IN {x.shape}')
        
        assert x.dtype == tf.int32
        x_bytes = tf.bitcast(x, tf.uint8)
        if self.verbose: print(x_bytes)
        x_float = tf.saturate_cast(x_bytes, self._target_dtype) / self._CONST_127 - self._ONE
        if self.verbose: print(f'x shape OUT {x_float.shape}')
        if self.reshape:
          shp = x_float.shape
          x_float = tf.reshape(x_float, shp[:-2] + [shp[-2]*4])
        return x_float
    
    def compute_output_shape(self, input_shape):
        shp = input_shape[:-1] + [4 * input_shape[-1]] if self.reshape else input_shape + [4]
        return shp

    def compute_output_signature(self, input_signature):
        return (tf.TensorSpec())

#    def build(self, input_shape):
#        #if self.reshape
#        self.output_dim = input_shape + [4]
#        super(Int32ToFloatLayer, self).build(input_shape) 


class FloatToInt32Layer(tf.keras.layers.Layer):
    def __init__(self,reshape: bool = False, nonlinearity_fn = tf.sigmoid, verbose=True,  **kwargs):
        super(FloatToInt32Layer, self).__init__(**kwargs)
        self.reshape = reshape
        self.nonlinearity_fn = nonlinearity_fn
        self.verbose=verbose
        self._CONST_127 = tf.constant(127.5, tf.float16) # TODO move from here


#    def build(self, input_shape):
#        self.output_dim = input_shape[1] // 4
#        super(FloatToInt32Layer, self).build(input_shape)

    def call(self, x):

        self._CONST_127 = tf.constant(127.5, x.dtype) # TODO move from here
        self._ONE = tf.constant(1.0, x.dtype) # TODO move from here
        
        if self.verbose: print(f'deconv bemenet {x}')
        x_out=x
        if self.nonlinearity_fn:
          x_out = self.nonlinearity_fn(x)
        if self.reshape:
          new_shape = x_out.shape
          new_shape = new_shape[:-1] + [new_shape[-1]//4] + [4]
          if self.verbose: print(f'reshaping from {x_out.shape} to {new_shape}')
          x_out = tf.reshape(x_out, new_shape)
        #TANH
        # Convert float16 to bytes
        #print(f'__1__ {x.shape}')
        if self.verbose: print(f'visszaszorozva {x_out * self._CONST_127}')
        x_bytes = tf.cast((x_out+self._ONE) * self._CONST_127, tf.uint8)
        if self.verbose: print(x_bytes)
        #print(f'__2__ {x_bytes.shape}')
        # Reshape the byte tensor to remove the byte dimension
        #x_bytes = tf.reshape(x_bytes, (-1, x.shape[1]))
        #print(f'__3__ {x_bytes.shape}')
        
        # Convert bytes to int32
        x_int32 = tf.bitcast(x_bytes, tf.int32)
        #print(f'__4__ {x_int32.shape}')
        if self.verbose: print(f'deconv kimenet {x_int32}')
        
        return x_int32
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 4)

