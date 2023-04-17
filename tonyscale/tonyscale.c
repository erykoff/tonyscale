/*
 *  Copyright (C) 2022 Eli Rykoff
 *  Author: Eli Rykoff
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

PyDoc_STRVAR(scale_image_doc,
             "scale_image(image, n_bins=100_000, n_colosr=256)\n"
             "--\n\n"
             "Scale an image the Tony Johnson way.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "image : `np.ndarray` (N, M)\n"
             "    Image to scale.\n"
             "n_bins : `int`, optional\n"
             "    Number of bins to use for histogram CDF.\n"
             "n_colors : `int`, optional\n"
             "    Number of colors to scale to.\n");

static PyObject *scale_image(PyObject *dummy, PyObject *args, PyObject *kwargs) {
    PyObject *image_obj = NULL;
    PyObject *image_arr = NULL;
    PyObject *scaled_image_arr = NULL;
    PyArrayIterObject *itr = NULL;
    long n_bins = 100000;
    long n_colors = 256;
    static char *kwlist[] = {"image", "n_bins", "n_colors", NULL};
    long *hist = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|LL", kwlist, &image_obj, &n_bins, &n_colors))
        goto fail;

    image_arr =
        PyArray_FROM_OTF(image_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);
    if (image_arr == NULL) goto fail;

    itr = (PyArrayIterObject *) PyArray_IterNew(image_arr);
    if (itr == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "Error creating iterator over image.");
        goto fail;
    }

    // Get min/max of image.
    double min = INFINITY;
    double max = -INFINITY;

    while (PyArray_ITER_NOTDONE(itr)) {
        double *val = (double * )PyArray_ITER_DATA(itr);

        if (*val < min) {
            min = *val;
        }
        if (*val > max) {
            max = *val;
        }
        PyArray_ITER_NEXT(itr);
    }
    PyArray_ITER_RESET(itr);

    // Make the histogram.
    long h_min = (long) (min - 1.0);
    long h_max = (long) (max + 1.0);
    double bin_size = ((double) h_max - (double) h_min)/(double)n_bins;

    if ((hist = (long *) calloc(n_bins, sizeof(long))) == NULL) {
        goto fail;
    }

    while (PyArray_ITER_NOTDONE(itr)) {
        double *val = (double *)PyArray_ITER_DATA(itr);
        long index = (long) ((*val - (double) h_min)/ (double) bin_size);

        hist[index] += 1;

        PyArray_ITER_NEXT(itr);
    }
    PyArray_ITER_RESET(itr);

    npy_intp len = PyArray_SIZE((PyArrayObject *)image_arr);

    // Change the histogram to a cumulative CDF.
    for (long i = 1; i<n_bins; i++) {
        hist[i] += hist[i - 1];
    }

    // Compute the CDF in the histogram.
    for (long i = 0; i<n_bins; i++) {
        hist[i] = (long) ((n_colors - 1)*hist[i]/len);
    }

    // Create the output image.
    scaled_image_arr = PyArray_SimpleNew(PyArray_NDIM((PyArrayObject *)image_arr), PyArray_DIMS((PyArrayObject *)image_arr), NPY_INT64);
    if (scaled_image_arr == NULL) goto fail;
    int64_t *scaled_image_data = (int64_t *)PyArray_DATA((PyArrayObject *)scaled_image_arr);

    while (PyArray_ITER_NOTDONE(itr)) {
        double *val = (double *)PyArray_ITER_DATA(itr);
        long index = (long) ((*val - (double) h_min)/ (double) bin_size);

        scaled_image_data[itr->index] = hist[index];

        PyArray_ITER_NEXT(itr);
    }

    Py_DECREF(image_arr);
    Py_DECREF(itr);
    free(hist);

    return PyArray_Return((PyArrayObject *) scaled_image_arr);
 fail:
    Py_XDECREF(image_arr);
    Py_XDECREF(scaled_image_arr);
    Py_XDECREF(itr);
    if (hist != NULL) {
        free(hist);
    }

    return NULL;
}

static PyMethodDef tonyscale_methods[] = {
    {"scale_image", (PyCFunction)(void (*)(void))scale_image,
     METH_VARARGS | METH_KEYWORDS, scale_image_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef tonyscale_module = {PyModuleDef_HEAD_INIT, "_tonyscale", NULL, -1,
                                              tonyscale_methods};

PyMODINIT_FUNC PyInit__tonyscale(void) {
    import_array();
    return PyModule_Create(&tonyscale_module);
}
