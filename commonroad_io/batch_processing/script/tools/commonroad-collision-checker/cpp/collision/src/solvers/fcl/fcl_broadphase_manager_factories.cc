/*
 * fcl_broadphase_manager_factories.cc
 *
 *  Created on: Jun 10, 2018
 *  Author: Vitaliy Rusinov
 */
#include "collision/solvers/fcl/fcl_broadphase_manager_factory.h"
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>

namespace collision {
namespace solvers {
namespace solverFCL {
BroadPhaseManagerFactoryConstPtr defaultFactory;

void initialize_default_broadphase_factory(void) {
  defaultFactory = std::shared_ptr<BroadPhaseManagerFactory>(
      (BroadPhaseManagerFactory *)(new DynamicAABBTreeManagerFactory()));
}

BroadPhaseManagerFactoryConstPtr getDefaultBroadphaseFactory(void)

{
  static bool broadPhaseDefaultFactoryInitialized = false;
  if (!broadPhaseDefaultFactoryInitialized) {
    initialize_default_broadphase_factory();
    broadPhaseDefaultFactoryInitialized = true;
  }
  return defaultFactory;
}

fcl::BroadPhaseCollisionManager<FCL_PRECISION> *
DynamicAABBTreeManagerFactory::instantiateBroadphaseManager(void) const {
  return new fcl::DynamicAABBTreeCollisionManager<FCL_PRECISION>();
}

} // namespace solverFCL
} // namespace solvers
} // namespace collision
