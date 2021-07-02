import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, enroll_path, test_path):
    pass

