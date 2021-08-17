#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

extern "C" {

// Naive implementation of convolution 2d layer in C++ (Conv2D + BiasAdd + ReLU)
// Param : strides=(1,1), padding='valid', kernel_size=(3,3), filters=64
static PyObject* conv2d(PyObject* self, PyObject* args) {

    // Parse arguments
    PyArrayObject *input, *weight, *bias;
    PyObject *output;
    if(!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &input, &PyArray_Type, &weight, &PyArray_Type, &bias)) { // Arguments are Numpy array objects ("O!")
        return nullptr;
    }

    // Obtain data buffer pointers and some attributes of Numpy objects
    float *input_buf  = static_cast<float*>(PyArray_DATA(input));         // PyArray_DATA() will return void*
    float *weight_buf = static_cast<float*>(PyArray_DATA(weight));
    float *bias_buf   = static_cast<float*>(PyArray_DATA(bias));
    size_t ndim            = PyArray_NDIM(input);                         // Number of dimensions of the input Numpy array
    npy_intp *input_shape  = PyArray_SHAPE(input);                        // Shape of the input Numpy array (npy_intp[])
    npy_intp *weight_shape = PyArray_SHAPE(weight);                       // Convolution kernel shape
    npy_intp *bias_shape   = PyArray_SHAPE(bias);

    //std::cout << input_shape[0] << "," << input_shape[1] << "," << input_shape[2] << "," << input_shape[3] << std::endl;     // 1,64,5,5   n,c,h,w
    //std::cout << weight_shape[0] << "," << weight_shape[1] << "," << weight_shape[2] << "," << weight_shape[3] << std::endl; // 3,3,64,64  h,w,c,n
    //std::cout << bias_shape[0] << std::endl;                                                                                 // 64

    size_t input_ch      = input_shape[1];
    size_t input_width   = input_shape[3];             // image height
    size_t input_height  = input_shape[2];             // image width
    size_t weight_height = weight_shape[0];
    size_t weight_width  = weight_shape[1];
    size_t weight_ch     = weight_shape[2];
    size_t weight_num    = weight_shape[3];

    // Create a Numpy object to store result
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT32);
    size_t output_width  = input_width-weight_width+1;
    size_t output_height = input_height-weight_height+1; 
    std::vector<npy_intp> output_shape {1, (npy_intp)weight_num, (npy_intp)output_width, (npy_intp)output_height};              // Shape
    output = PyArray_Zeros(output_shape.size(), output_shape.data(), descr, 0);                                                 // 1,64,3,3  n,c,h,w
    float* output_buf = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(output)));                            // Obtain pointer to the data

    size_t kernel_size = weight_width;  // 3
    size_t stride = 1;
    size_t filters = weight_num;        // 64

    // Conv2D + BiasAdd + ReLU main loop
    for(size_t fc=0; fc<filters; fc++) {
        for(size_t dy=0; dy<output_height; dy+=stride) {
            for(size_t dx=0; dx<output_width; dx+=stride) {
                float cnv = 0.f;
                for(size_t cc=0; cc<input_ch; cc++) {
                    for(size_t fy=0; fy<weight_height; fy++) {
                        for(size_t fx=0; fx<weight_width; fx++) {
                            float flt = weight_buf[fc + cc*(filters) + fx*(filters*input_ch) + fy*(filters*input_ch*weight_width)];
                            float dt  = input_buf[(dx+fx) + (dy+fy)*(input_width) + cc*(input_width*input_height) ];
                            cnv += flt * dt;
                        }
                    }
                }
                cnv += bias_buf[fc];                 // BiasAdd
                cnv = cnv < 0.f ? 0.f : cnv;         // ReLU
                output_buf[dx + dy*(output_width) + fc*(output_width*output_height)] = cnv;
            }
        }
    }
    return output;
}


//---------------------------------------------------------------------------------------------

// Function definition table to export to Python
PyMethodDef method_table[] = {
    {"conv2d", static_cast<PyCFunction>(conv2d), METH_VARARGS, "Conv2D+BiasAdd+ReLU"},
    {NULL, NULL, 0, NULL}
};

// Module definition table
PyModuleDef myLayers_module = {
    PyModuleDef_HEAD_INIT,
    "myLayers",
    "C++ Custom Layer Implementation",
    0,
    method_table
};

// Initialize and register module function
// Function name must be 'PyInit_'+module name
// This function must be the only *non-static* function in the source code
PyMODINIT_FUNC PyInit_myLayers(void) {
    import_array();                                 // Required to receive Numpy object as arguments
    if (PyErr_Occurred()) {
        return nullptr;
    }
    return PyModule_Create(&myLayers_module);
}

}; // extern "C"