from intlayer.intlayer import Int32ToFloatLayer, FloatToInt32Layer
import numpy as np
import itertools
import sys
import tensorflow as tf

if 'ipykernel' in sys.modules:
    # Running in a Jupyter notebook
    from tqdm.notebook import tqdm
else:
    # Not running in a Jupyter notebook
    from tqdm import tqdm


def run_test_int32_to_floats(P, chunk_size=2**24):
    pbar = tqdm(total=2**25)
    int32_to_float16_layer = Int32ToFloatLayer(verbose=False, **P)
    float16_to_int32_layer = FloatToInt32Layer(verbose=False, 
                                               **P,  
                                               nonlinearity_fn=None)

    # for i in range(tf.int32.min, tf.int32.max, chunk_size):

    for i in range(tf.int32.min, tf.int32.min + chunk_size, chunk_size):
    
        x_chunk = tf.range(start=i, 
                           limit=min(i+chunk_size, tf.int32.max), 
                           dtype=tf.int32)
        f_chunk1 = int32_to_float16_layer(x_chunk)
        x_chunk2 = float16_to_int32_layer(f_chunk1)
        assert np.all(x_chunk == x_chunk2), (
            f"Conversion failed for chunk starting at x = {i}")
        pbar.update(chunk_size)
    pbar.close()
    print("All conversions successful")


def test_long_test():
    param_dict = {
        'reshape': [True, False],
        'dtype': Int32ToFloatLayer.TESTED_DTYPES,
        'dims': [1, 2, 3, 4, 5]
    }
    param_generator = (dict(zip(param_dict, x)) 
                       for x in itertools.product(*param_dict.values()))

    for P in list(param_generator):
        print(P)
        P.pop('dims')
        run_test_int32_to_floats(P)


def test_always_pass():
    pass
