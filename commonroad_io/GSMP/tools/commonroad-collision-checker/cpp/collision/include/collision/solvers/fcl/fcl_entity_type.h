/*
 * fcl_entity_type.h
 *
 *  Created on: May 27, 2018
 *  Author: Vitaliy Rusinov
 */

#ifndef CPP_COLLISION_FCL_FCL_ENTITY_TYPE_H_
#define CPP_COLLISION_FCL_FCL_ENTITY_TYPE_H_

namespace collision {
namespace solvers {
namespace solverFCL {
enum FCL_COLLISION_ENTITY_TYPE {
  FCL_COLLISION_ENTITY_TYPE_INVALID = -1,
  FCL_COLLISION_ENTITY_TYPE_UNKNOWN = 0,
  COLLISION_ENTITY_TYPE_FCL_OBJECT = 200,
  COLLISION_ENTITY_TYPE_FCL_OBJECTGROUP = 201,
  FCL_COLLISION_ENTITY_TYPE_TVOBJECT = 202
};
}
} // namespace solvers
} // namespace collision

#endif /* CPP_COLLISION_FCL_FCL_ENTITY_TYPE_H_ */
