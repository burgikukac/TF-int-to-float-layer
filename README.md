# Large int inputs in tensorflow.


This repository contains two custom Keras layers for TensorFlow that convert between `Int32` and `float` tensors:

`Int32ToFloatLayer`: A layer that converts `Int32` tensors to `float` tensors.
`FloatToInt32Layer`: A layer that converts `float` tensors to `Int32` tensors.

The rationale behind is to keep all of the bit information of potentially
  very big integer inputs for a deep learning model. This could be achieved by
  simply converting the numbers to `Float64` and rescaling, however recent
  architectures tend to use less precision.
  Converting an `Int32` to 32 floats bit-by-bit seems to be too many.
  This solution is to convert an `Int32` to 4 floats.

  We take the input integers(for instance an `Int32`), and byte-by-byte
  convert them to floats.

My use case was sorting a vector of large integers, where I should be able to test
whether I got back the exact answer, not just an approximation.

  __These layers intended to being used only in experimentation.__



To use these layers in your TensorFlow models, simply import them from the intlayer module and add them to your model as you would any other Keras layer.