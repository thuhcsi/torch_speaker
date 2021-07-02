import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger()

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, enroll_path, test_path):
    # Create an execution context and specify input shape (based on the image dimensions for inference).
    # Allocate CUDA device memory for input and output.
    # Allocate CUDA page-locked host memory to efficiently copy back the output.
    # Transfer the processed image data into input memory using asynchronous host-to-device CUDA copy.
    # Kickoff the TensorRT inference pipeline using the asynchronous execute API.
    # Transfer the segmentation output back into pagelocked host memory using device-to-host CUDA copy.
    # Synchronize the stream used for data transfers and inference execution to ensure all operations are completes.
    pass

