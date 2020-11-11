//*  Author: Vitaliy Rusinov

#pragma once
#include "collision/collision_object.h"
#include "collision/i_collision_container.h"
#include "collision/solvers/fcl/i_solver_entity_fcl.h"

namespace collision {
using namespace solvers::solverFCL;

namespace solvers {
namespace solverFCL {
class SolverEntity_FCL;
struct SolverEntity_FCLDeleter {
  void operator()(SolverEntity_FCL *p);
};
} // namespace solverFCL
} // namespace solvers

/*!
\brief Provides functionality to use different collision solvers

*/
class CollisionObjectEx : public collision::CollisionObject {

public:
  CollisionObjectEx() { fcl_solver_entity_valid_ = false; }
  virtual ~CollisionObjectEx() {}

  virtual bool
  collide(const CollisionObject &c,
          const collision::CollisionRequest &req = CollisionRequest()) const;

  virtual bool BVCheck(CollisionObjectConstPtr obj2) const;

  virtual std::shared_ptr<const collision::RectangleAABB> getAABB() const;

  virtual int getSolverEntity(SolverEntity_FCL *&ptr) const;

  virtual const ICollisionContainer *getContainerInterface(void) const {
    return nullptr;
  }

  virtual const solvers::solverFCL::ISolverEntity_FCL *
  getFclInterface(void) const {
    return nullptr;
  }

protected:
  void invalidateCollisionEntityCache(void);

private:
  mutable std::unique_ptr<SolverEntity_FCL, SolverEntity_FCLDeleter>
      fcl_entity_;
  mutable bool fcl_solver_entity_valid_;
};
} // namespace collision
