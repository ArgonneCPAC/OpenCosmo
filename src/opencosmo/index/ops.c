#include <Python.h> // clang-format off
//
#include "methodobject.h"
#include "numpy/ndarrayobject.h"
#include "tupleobject.h"


static PyObject* method_chunked_into_array(PyObject *self, PyObject *args) {
	PyArrayObject *start, *size;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &start, &PyArray_Type, &size)) {
		return NULL;
	}
	if (!PyArray_EquivTypenums(PyArray_TYPE(start), NPY_INT64) || !PyArray_EquivTypenums(PyArray_TYPE(size), NPY_INT64 )) {
		PyErr_SetString(PyExc_ValueError, "Index must contain integers");
		return NULL;
	}
	if (!PyArray_IS_C_CONTIGUOUS(start) || !PyArray_IS_C_CONTIGUOUS(size)) {
		PyErr_SetString(PyExc_ValueError, "Index must be contiguous!");
		return NULL;
	}


	npy_intp length_ = PyArray_DIM(start, 0);
	npy_int64 *size_data = (npy_int64*)PyArray_DATA(size);
	npy_int64 *start_data = (npy_int64*)PyArray_DATA(start);
	npy_int64 total_size = 0;

	for (npy_intp i = 0; i < length_; i++ ) {
		total_size += size_data[i];
	}

	npy_intp dims[1] = {total_size};
	PyObject *out_arr = PyArray_SimpleNew(1, dims, NPY_INT64);
	if (out_arr == NULL ){
		return NULL;
	}

	npy_int64 *out_data = (npy_int64*)PyArray_DATA((PyArrayObject*)out_arr);

	npy_int64 rs = 0;
	for (npy_intp i = 0; i < length_; i ++ ) {
		npy_int64 chunk_size = size_data[i];
		npy_int64 chunk_start = start_data[i];
		for (npy_int64 j = 0; j < chunk_size; j++) {
			out_data[rs + j] = chunk_start+j;
		}
		rs += chunk_size;
	}

	return out_arr;
}

static PyObject* method_get_simple_range(PyObject* self, PyObject *args) {
	PyArrayObject *index;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &index)) {
		return NULL;
	}
	if (!PyArray_EquivTypenums(PyArray_TYPE(index), NPY_INT64)) {
		PyErr_SetString(PyExc_ValueError, "Index must contain integers");
		return NULL;
	}
	if (!PyArray_IS_C_CONTIGUOUS(index)) {
		PyErr_SetString(PyExc_ValueError, "Index must be contiguous!");
		return NULL;
	}
	PyObject* size_out = PyTuple_New(2);
	if (PyArray_DIM(index, 0) == (npy_intp)0) {
		PyTuple_SET_ITEM(size_out, 0, PyLong_FromLong(0));
		PyTuple_SET_ITEM(size_out, 1, PyLong_FromLong(0));
		return size_out;
	}

	npy_intp length = PyArray_DIM(index, 0);
	npy_int64 *index_data = (npy_int64*)PyArray_DATA(index);
	npy_int64 min = index_data[0];
	npy_int64 max = index_data[0];
	
	for (npy_intp i = 1; i < length; i++) {
		npy_int64 value = index_data[i];
		if (value < min) {
			min = value;
		}
		if (value > max) {
			max = value;
		}
	}
	PyTuple_SET_ITEM(size_out, 0, PyLong_FromLongLong(min));
	PyTuple_SET_ITEM(size_out, 1, PyLong_FromLongLong(max));
	return size_out;
}

static PyMethodDef _opsMethods[] = {
    {"chunked_into_array", method_chunked_into_array, METH_VARARGS,
     "Convert a chunked index into an array"},
    {"get_simple_range", method_get_simple_range, METH_VARARGS, "Determine the range of a simple index"},

    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _opsModule = {
    PyModuleDef_HEAD_INIT, "_ops",
    "Python interface for the fputs C library function", -1, _opsMethods};

PyMODINIT_FUNC PyInit__ops(void) {
import_array();
return PyModule_Create(&_opsModule);
}
