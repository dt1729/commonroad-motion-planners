#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef PY_WRAPPER_MODULE_COLLISION
void init_module_collision(py::module &m);
#endif

PYBIND11_MODULE(pycrcc, m) {

    #ifdef PY_WRAPPER_MODULE_COLLISION
    init_module_collision(m);
    #endif
}
