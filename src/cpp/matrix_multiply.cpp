#include <Python.h>
#include <numpy/arrayobject.h>

// Matrix multiplication function
extern "C" {
    void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                C[i * K + j] = 0;
                for (int k = 0; k < N; ++k) {
                    C[i * K + j] += A[i * N + k] * B[k * K + j];
                }
            }
        }
    }
}

// Python wrapper function
static PyObject* py_matrix_multiply(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;

    // Parse the arguments (A, B)
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a_obj, &PyArray_Type, &b_obj)) {
        return NULL;
    }

    // Convert Python objects (arrays) to NumPy arrays
    PyArrayObject* A = (PyArrayObject*)a_obj;
    PyArrayObject* B = (PyArrayObject*)b_obj;

    // Ensure arrays have the correct dtype (float32) and are contiguous in memory
    if (PyArray_TYPE(A) != NPY_FLOAT32 || PyArray_TYPE(B) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "Arrays must be of type float32");
        return NULL;
    }
    if (!PyArray_ISCARRAY(A) || !PyArray_ISCARRAY(B)) {
        PyErr_SetString(PyExc_TypeError, "Arrays must be contiguous");
        return NULL;
    }

    // Get dimensions programmatically
    int M = PyArray_DIM(A, 0);  // Number of rows in A
    int N = PyArray_DIM(A, 1);  // Number of columns in A, should match rows in B
    int K = PyArray_DIM(B, 1);  // Number of columns in B

    // Validate dimensions for matrix multiplication
    if (N != PyArray_DIM(B, 0)) {
        PyErr_SetString(PyExc_ValueError, "Matrix A's columns must match Matrix B's rows for multiplication");
        return NULL;
    }

    // Create output array C with shape (M, K)
    npy_intp dims[2] = {M, K};
    PyArrayObject* C = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_FLOAT32, 0);
    if (!C) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for output array C");
        return NULL;
    }

    // Call the C++ matrix_multiply function
    matrix_multiply(
        (float*)PyArray_DATA(A),  // Pointer to A
        (float*)PyArray_DATA(B),  // Pointer to B
        (float*)PyArray_DATA(C),  // Pointer to C
        M, N, K
    );

    return Py_BuildValue("O", C); // Return the result array C
}

// Method definition table
static PyMethodDef MatrixMethods[] = {
    {"matrix_multiply", py_matrix_multiply, METH_VARARGS, "Matrix multiplication"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef matrix_module = {
    PyModuleDef_HEAD_INIT,
    "matrix_module",   // Module name
    "Matrix Multiplication Module",  // Module description
    -1,                // Size of per-interpreter state of the module
    MatrixMethods      // Method table
};

// Module initialization
PyMODINIT_FUNC PyInit_mx(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&matrix_module);
}
