set(COLLISION_SRC_ROOT "collision/src")

set(COLLISION_SRC 	
${COLLISION_SRC_ROOT}/raytrace_primitive.cc
${COLLISION_SRC_ROOT}/shape_group.cc
${COLLISION_SRC_ROOT}/collision_object.cc
${COLLISION_SRC_ROOT}/collision_checker.cc
${COLLISION_SRC_ROOT}/primitive_collision_checker.cc
${COLLISION_SRC_ROOT}/raytrace_utils.cc
${COLLISION_SRC_ROOT}/collision_object_ex.cc
${COLLISION_SRC_ROOT}/time_variant_collision_object.cc
${COLLISION_SRC_ROOT}/solvers/distance_queries.cc
${COLLISION_SRC_ROOT}/solvers/collision_queries.cc
${COLLISION_SRC_ROOT}/solvers/collision_solver_default.cc
${COLLISION_SRC_ROOT}/solvers/collision_solver_fcl.cc
${COLLISION_SRC_ROOT}/solvers/collision_solver_primitive.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_collision_checker.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_distance_queries.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_broadphase_manager_factories.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_distance_requests.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_collision_queries.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_collision_object_group.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_collision_requests.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_collision_object.cc
${COLLISION_SRC_ROOT}/solvers/fcl/fcl_entity_factory.cc
${COLLISION_SRC_ROOT}/solvers/fcl/performance_timers.cc
${COLLISION_SRC_ROOT}/solvers/fcl/solver_entity_fcl.cc
${COLLISION_SRC_ROOT}/plugins/triangulation/triangulate.cc
${COLLISION_SRC_ROOT}/serialize/serialize_rectangle_aabb.cc
${COLLISION_SRC_ROOT}/serialize/serialize_collision_checker.cc
${COLLISION_SRC_ROOT}/serialize/serialize_rectangle_obb.cc
${COLLISION_SRC_ROOT}/serialize/serialize_tv_object.cc
${COLLISION_SRC_ROOT}/serialize/serialize_point.cc
${COLLISION_SRC_ROOT}/serialize/serialize_triangle.cc
${COLLISION_SRC_ROOT}/serialize/serialize_container_cc_obj.cc
${COLLISION_SRC_ROOT}/serialize/serialize_container_obj_obj.cc
${COLLISION_SRC_ROOT}/serialize/serialize_polygon.cc
${COLLISION_SRC_ROOT}/serialize/serialize_shape_group.cc
${COLLISION_SRC_ROOT}/serialize/serialize.cc
${COLLISION_SRC_ROOT}/serialize/serialize_sphere.cc
${COLLISION_SRC_ROOT}/serialize/public/serialize_public.cc
${COLLISION_SRC_ROOT}/serialize/final/collision_object_export_final.cc
${COLLISION_SRC_ROOT}/narrowphase/polygon.cc
${COLLISION_SRC_ROOT}/narrowphase/rectangle_aabb.cc
${COLLISION_SRC_ROOT}/narrowphase/triangle.cc
${COLLISION_SRC_ROOT}/narrowphase/rectangle_obb.cc
${COLLISION_SRC_ROOT}/narrowphase/shape.cc
${COLLISION_SRC_ROOT}/narrowphase/point.cc
${COLLISION_SRC_ROOT}/narrowphase/sphere.cc
tests/collision/online_tests/broadphase_test.cc
tests/collision/online_tests/test_common.cc
)


set(LIB_CRCC_SRC ${LIB_CRCC_SRC} ${COLLISION_SRC})

find_package(fcl REQUIRED)

find_package(ccd REQUIRED)

find_library(S11N_LIBRARY s11n HINTS ${PC_LIBS11N_LIBRARIES})

if(NOT fcl_FOUND)
    message(FATAL_ERROR "Could NOT find required library libfcl")
endif()

add_library(crcc ${LIB_CRCC_SRC})

target_compile_features(crcc
    PUBLIC cxx_lambdas cxx_auto_type
    PRIVATE cxx_lambdas cxx_auto_type)
set_property(TARGET crcc PROPERTY POSITION_INDEPENDENT_CODE ON)
set(COLLISION_INCLUDES "./collision/include" ".")
target_include_directories(crcc PUBLIC ${COLLISION_INCLUDES})


if(NOT S11N_LIBRARY)
	set(S11N_LIBRARY "")
	add_definitions(-DENABLE_SERIALIZER=0)
endif()
target_link_libraries(crcc PUBLIC ccd fcl ${S11N_LIBRARY})


