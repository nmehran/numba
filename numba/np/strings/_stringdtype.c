#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#define NUMBA_NUMPY_2_0_API_VERSION 0x00000012

/* NumPy StringDType C-API slots used by Numba's native helper. */
#define NUMBA_NPYSTRING_ACQUIRE_ALLOCATOR_SLOT 316
#define NUMBA_NPYSTRING_ACQUIRE_ALLOCATORS_SLOT 317
#define NUMBA_NPYSTRING_RELEASE_ALLOCATORS_SLOT 319

#if NPY_API_VERSION >= NUMBA_NUMPY_2_0_API_VERSION
#define NUMBA_HAS_STRINGDTYPE_API 1
#else
#define NUMBA_HAS_STRINGDTYPE_API 0
#endif

void *
numba_stringdtype_acquire_allocator(PyObject *array)
{
#if NUMBA_HAS_STRINGDTYPE_API
    PyArray_Descr *descr = PyArray_DESCR((PyArrayObject *)array);
    typedef void *(*acquire_allocator_func)(const void *);
    acquire_allocator_func acquire = (acquire_allocator_func)
        PyArray_API[NUMBA_NPYSTRING_ACQUIRE_ALLOCATOR_SLOT];
    return acquire((const void *)descr);
#else
    (void)array;
    return NULL;
#endif
}


void
numba_stringdtype_acquire_two_allocators(
    PyObject *left,
    PyObject *right,
    void **allocators
)
{
#if NUMBA_HAS_STRINGDTYPE_API
    PyArray_Descr *descrs[2] = {
        PyArray_DESCR((PyArrayObject *)left),
        PyArray_DESCR((PyArrayObject *)right),
    };
    typedef void (*acquire_allocators_func)(
        size_t,
        PyArray_Descr *const[],
        void **
    );
    acquire_allocators_func acquire = (acquire_allocators_func)
        PyArray_API[NUMBA_NPYSTRING_ACQUIRE_ALLOCATORS_SLOT];
    acquire(2, descrs, allocators);
#else
    (void)left;
    (void)right;
    allocators[0] = NULL;
    allocators[1] = NULL;
#endif
}


void
numba_stringdtype_release_two_allocators(void **allocators)
{
#if NUMBA_HAS_STRINGDTYPE_API
    typedef void (*release_allocators_func)(size_t, void **);
    release_allocators_func release = (release_allocators_func)
        PyArray_API[NUMBA_NPYSTRING_RELEASE_ALLOCATORS_SLOT];
    release(2, allocators);
#else
    (void)allocators;
#endif
}


static PyObject *
has_stringdtype_api(PyObject *self, PyObject *args)
{
    (void)self;
    (void)args;
#if NUMBA_HAS_STRINGDTYPE_API
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}


static PyMethodDef methods[] = {
    {
        "has_stringdtype_api",
        has_stringdtype_api,
        METH_NOARGS,
        "Return whether this helper was built with NumPy StringDType API.",
    },
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_stringdtype",
    NULL,
    -1,
    methods,
};


PyMODINIT_FUNC
PyInit__stringdtype(void)
{
    import_array();
    return PyModule_Create(&module);
}
