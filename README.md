# Large int inputs in Tensorflow.


This repository contains two custom Keras layers for TensorFlow that convert between `Int32` and `float` tensors:

- **Int32ToFloatLayer**: A layer that converts `Int32` tensors to `float` tensors.  
- **FloatToInt32Layer**: A layer that converts `float` tensors to `Int32` tensors.

The rationale behind these layers is to retain all the bit-level information of potentially very large integer inputs in a deep learning model. While this could be achieved by simply converting the numbers to `Float64` and rescaling, recent architectures tend to use lower precision. Converting an `Int32` to 32 floats bit-by-bit seems excessive, so the solution is to convert an `Int32` into 4 floats instead.

The method involves taking input integers (e.g., an `Int32`) and converting them byte-by-byte into floats.

My use case involved sorting a vector of large integers, where it was essential to verify that the exact values were returned, not just an approximation.

These layers are intended for experimental use only.

To use these layers in your TensorFlow models, simply import them from the `intlayer` module and add them to your model like any other Keras layer.
