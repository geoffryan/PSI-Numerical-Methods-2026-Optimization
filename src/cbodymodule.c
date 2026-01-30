#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#include "mycalc.h"

static PyObject * cbody_calc_g(PyObject *self, PyObject *args);
static PyObject * cbody_calc_en(PyObject *self, PyObject *args);
static PyObject * cbody_calc_g_par(PyObject *self, PyObject *args);
static PyObject * cbody_calc_en_par(PyObject *self, PyObject *args);

static PyMethodDef cbody_methods[] = {
    {"calc_g", cbody_calc_g, METH_VARARGS,
        "Calculate acceleration due to gravity."},
    {"calc_g_par", cbody_calc_g_par, METH_VARARGS,
        "Calculate acceleration due to gravity in parallel."},
    {"calc_en", cbody_calc_en, METH_VARARGS,
        "Calculate acceleration due to gravity."},
    {"calc_en_par", cbody_calc_en_par, METH_VARARGS,
        "Calculate acceleration due to gravity in parallel."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cbody_module = {
    .m_methods = cbody_methods,
};

PyMODINIT_FUNC PyInit_cbody(void)
{
    import_array();

    return PyModuleDef_Init(&cbody_module);
}

static PyObject * cbody_calc_g(PyObject *self, PyObject *args)
{
    PyObject *g_obj = NULL;
    PyObject *r_obj = NULL;
    PyObject *m_obj = NULL;
    double eps_soft = 0.0;

    if(!PyArg_ParseTuple(args, "OOOd", &g_obj, &r_obj, &m_obj, &eps_soft)) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }
    PyArrayObject *g_arr;
    PyArrayObject *r_arr;
    PyArrayObject *m_arr;

    g_arr = (PyArrayObject *) PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_OUT_ARRAY);
    r_arr = (PyArrayObject *) PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);
    m_arr = (PyArrayObject *) PyArray_FROM_OTF(m_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);

    if(g_arr == NULL || r_arr == NULL || m_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(g_arr);
        Py_XDECREF(r_arr);
        Py_XDECREF(m_arr);
        return NULL;
    }
    
    int g_ndim = (int) PyArray_NDIM(g_arr);
    int r_ndim = (int) PyArray_NDIM(r_arr);
    int m_ndim = (int) PyArray_NDIM(m_arr);

    if(g_ndim != 2 || r_ndim != 2 || m_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "g and r must be 2D and m must be 1D");
        Py_DECREF(g_arr);
        Py_DECREF(r_arr);
        Py_DECREF(m_arr);
        return NULL;
    }

    long ncomp = (long)PyArray_DIM(r_arr, 1);
    long N = (long)PyArray_DIM(r_arr, 0);
    long ncomp2 = (long)PyArray_DIM(g_arr, 1);
    long N2 = (long)PyArray_DIM(g_arr, 0);
    long N3 = (long)PyArray_DIM(m_arr, 0);

    if(ncomp != 3 || ncomp2 != 3 || N2 != N || N3 != N)
    {
        PyErr_SetString(PyExc_RuntimeError, "g and r must be [N, 3] and m must be [N]");
        Py_DECREF(g_arr);
        Py_DECREF(r_arr);
        Py_DECREF(m_arr);
        return NULL;
    }
    
    double *g = (double *)PyArray_DATA(g_arr);
    double *r = (double *)PyArray_DATA(r_arr);
    double *m = (double *)PyArray_DATA(m_arr);
    
    calc_g(g, r, m, N, eps_soft);
    
    Py_RETURN_NONE;
}

static PyObject * cbody_calc_en(PyObject *self, PyObject *args)
{
    PyObject *r_obj = NULL;
    PyObject *v_obj = NULL;
    PyObject *m_obj = NULL;
    double eps_soft = 0.0;

    if(!PyArg_ParseTuple(args, "OOOd", &r_obj, &v_obj, &m_obj, &eps_soft)) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }
    PyArrayObject *r_arr;
    PyArrayObject *v_arr;
    PyArrayObject *m_arr;

    r_arr = (PyArrayObject *) PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_OUT_ARRAY);
    v_arr = (PyArrayObject *) PyArray_FROM_OTF(v_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);
    m_arr = (PyArrayObject *) PyArray_FROM_OTF(m_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);

    if(r_arr == NULL || v_arr == NULL || m_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(r_arr);
        Py_XDECREF(v_arr);
        Py_XDECREF(m_arr);
        return NULL;
    }
    
    int r_ndim = (int) PyArray_NDIM(r_arr);
    int v_ndim = (int) PyArray_NDIM(v_arr);
    int m_ndim = (int) PyArray_NDIM(m_arr);

    if(r_ndim != 2 || v_ndim != 2 || m_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "r and v must be 2D and m must be 1D");
        Py_DECREF(r_arr);
        Py_DECREF(v_arr);
        Py_DECREF(m_arr);
        return NULL;
    }

    long N = (long)PyArray_DIM(r_arr, 0);
    long ncomp = (long)PyArray_DIM(r_arr, 1);
    long N2 = (long)PyArray_DIM(v_arr, 0);
    long ncomp2 = (long)PyArray_DIM(v_arr, 1);
    long N3 = (long)PyArray_DIM(m_arr, 0);

    if(ncomp != 3 || ncomp2 != 3 || N2 != N || N3 != N)
    {
        PyErr_SetString(PyExc_RuntimeError, "r and v must be [N, 3] and m must be [N]");
        Py_DECREF(r_arr);
        Py_DECREF(v_arr);
        Py_DECREF(m_arr);
        return NULL;
    }
    
    double *r = (double *)PyArray_DATA(r_arr);
    double *v = (double *)PyArray_DATA(v_arr);
    double *m = (double *)PyArray_DATA(m_arr);
    
    double e = calc_en(r, v, m, N, eps_soft);
    
    return PyFloat_FromDouble(e);
}

static PyObject * cbody_calc_g_par(PyObject *self, PyObject *args)
{
    PyObject *g_obj = NULL;
    PyObject *r_obj = NULL;
    PyObject *m_obj = NULL;
    double eps_soft = 0.0;
    int num_workers = 0;

    if(!PyArg_ParseTuple(args, "OOOdd", &g_obj, &r_obj, &m_obj, &eps_soft,
                &num_workers)) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }

    if(num_workers <= 0)
        num_workers = 1;

    PyArrayObject *g_arr;
    PyArrayObject *r_arr;
    PyArrayObject *m_arr;

    g_arr = (PyArrayObject *) PyArray_FROM_OTF(g_obj, NPY_DOUBLE,
                                               NPY_ARRAY_OUT_ARRAY);
    r_arr = (PyArrayObject *) PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);
    m_arr = (PyArrayObject *) PyArray_FROM_OTF(m_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);

    if(g_arr == NULL || r_arr == NULL || m_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(g_arr);
        Py_XDECREF(r_arr);
        Py_XDECREF(m_arr);
        return NULL;
    }

    int g_ndim = (int) PyArray_NDIM(g_arr);
    int r_ndim = (int) PyArray_NDIM(r_arr);
    int m_ndim = (int) PyArray_NDIM(m_arr);

    if(g_ndim != 2 || r_ndim != 2 || m_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "g and r must be 2D and m must be 1D");
        Py_DECREF(g_arr);
        Py_DECREF(r_arr);
        Py_DECREF(m_arr);
        return NULL;
    }

    long ncomp = (long)PyArray_DIM(r_arr, 1);
    long N = (long)PyArray_DIM(r_arr, 0);
    long ncomp2 = (long)PyArray_DIM(g_arr, 1);
    long N2 = (long)PyArray_DIM(g_arr, 0);
    long N3 = (long)PyArray_DIM(m_arr, 0);

    if(ncomp != 3 || ncomp2 != 3 || N2 != N || N3 != N)
    {
        PyErr_SetString(PyExc_RuntimeError, "g and r must be [N, 3] and m must be [N]");
        Py_DECREF(g_arr);
        Py_DECREF(r_arr);
        Py_DECREF(m_arr);
        return NULL;
    }

    double *g = (double *)PyArray_DATA(g_arr);
    double *r = (double *)PyArray_DATA(r_arr);
    double *m = (double *)PyArray_DATA(m_arr);

    calc_g_par(g, r, m, N, eps_soft, num_workers);

    Py_RETURN_NONE;
}

static PyObject * cbody_calc_en_par(PyObject *self, PyObject *args)
{
    PyObject *r_obj = NULL;
    PyObject *v_obj = NULL;
    PyObject *m_obj = NULL;
    double eps_soft = 0.0;
    int num_workers = 0;

    if(!PyArg_ParseTuple(args, "OOOdd", &r_obj, &v_obj, &m_obj, &eps_soft,
                &num_workers)) {
        PyErr_SetString(PyExc_RuntimeError, "Could not parse arguments.");
        return NULL;
    }

    if(num_workers <= 0)
        num_workers = 1;

    PyArrayObject *r_arr;
    PyArrayObject *v_arr;
    PyArrayObject *m_arr;

    r_arr = (PyArrayObject *) PyArray_FROM_OTF(r_obj, NPY_DOUBLE,
                                               NPY_ARRAY_OUT_ARRAY);
    v_arr = (PyArrayObject *) PyArray_FROM_OTF(v_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);
    m_arr = (PyArrayObject *) PyArray_FROM_OTF(m_obj, NPY_DOUBLE,
                                               NPY_ARRAY_IN_ARRAY);

    if(r_arr == NULL || v_arr == NULL || m_arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not read input arrays.");
        Py_XDECREF(r_arr);
        Py_XDECREF(v_arr);
        Py_XDECREF(m_arr);
        return NULL;
    }
    
    int r_ndim = (int) PyArray_NDIM(r_arr);
    int v_ndim = (int) PyArray_NDIM(v_arr);
    int m_ndim = (int) PyArray_NDIM(m_arr);

    if(r_ndim != 2 || v_ndim != 2 || m_ndim != 1)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "r and v must be 2D and m must be 1D");
        Py_DECREF(r_arr);
        Py_DECREF(v_arr);
        Py_DECREF(m_arr);
        return NULL;
    }

    long N = (long)PyArray_DIM(r_arr, 0);
    long ncomp = (long)PyArray_DIM(r_arr, 1);
    long N2 = (long)PyArray_DIM(v_arr, 0);
    long ncomp2 = (long)PyArray_DIM(v_arr, 1);
    long N3 = (long)PyArray_DIM(m_arr, 0);

    if(ncomp != 3 || ncomp2 != 3 || N2 != N || N3 != N)
    {
        PyErr_SetString(PyExc_RuntimeError, "r and v must be [N, 3] and m must be [N]");
        Py_DECREF(r_arr);
        Py_DECREF(v_arr);
        Py_DECREF(m_arr);
        return NULL;
    }
    
    double *r = (double *)PyArray_DATA(r_arr);
    double *v = (double *)PyArray_DATA(v_arr);
    double *m = (double *)PyArray_DATA(m_arr);
    
    double e = calc_en_par(r, v, m, N, eps_soft, num_workers);
    
    return PyFloat_FromDouble(e);
}
