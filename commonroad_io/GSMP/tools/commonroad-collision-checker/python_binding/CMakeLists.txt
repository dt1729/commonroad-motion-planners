include_directories("./collision/include")
pybind11_add_module(pycrcc SHARED src/module_collision.cc src/py_bind.cc)
set_target_properties(pycrcc PROPERTIES COMPILE_DEFINITIONS "PY_WRAPPER_MODULE_COLLISION=1")

target_link_libraries(pycrcc PRIVATE crcc)


