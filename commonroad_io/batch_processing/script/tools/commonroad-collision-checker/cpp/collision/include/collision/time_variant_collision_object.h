#ifndef TIME_VARIANT_COLLISION_OBJECT_H_
#define TIME_VARIANT_COLLISION_OBJECT_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "collision/collision_object.h"
#include "collision/narrowphase/shape.h"
#include "collision/shape_group.h"

namespace collision {

/*!
  \brief TimeVariantCollisionObject can contain a different CollisionObject or
  ShapeGroup at each time step


*/
class TimeVariantCollisionObject : public CollisionObjectEx {
public:
  virtual ~TimeVariantCollisionObject() {}
  TimeVariantCollisionObject(int time_start_idx);
  CollisionObjectConstPtr getObstacleAtTime(int time_idx) const;
  int appendObstacle(CollisionObjectConstPtr obstacle);
  virtual CollisionObjectConstPtr
  timeSlice(int time_idx, CollisionObjectConstPtr shared_ptr_this) const;

  int time_start_idx() const;
  int time_end_idx() const;

  bool rayTrace(const Eigen::Vector2d &point1, const Eigen::Vector2d &point2,
                std::vector<LineSegment> &intersect) const;

  //! Print all parameters of the shape
  virtual void print(std::ostringstream &stream) const;

  virtual void addParentMap(
      std::unordered_map<const CollisionObject *,
                         std::list<CollisionObjectConstPtr>> &parent_map) const;

  virtual CollisionObjectType getCollisionObjectType() const {
    return CollisionObjectType::OBJ_TYPE_TVOBSTACLE;
  }
  CollisionObjectClass getCollisionObjectClass() const {
    return CollisionObjectClass::OBJ_CLASS_TVOBSTACLE;
  }

#if ENABLE_SERIALIZER
  serialize::ICollisionObjectExport *exportThis(void) const;
#endif

protected:
  int time_start_idx_;
  int time_end_idx_;
  std::vector<CollisionObjectConstPtr> collision_object_at_time_;
};

typedef std::shared_ptr<TimeVariantCollisionObject>
    TimeVariantCollisionObjectPtr;
typedef std::shared_ptr<const TimeVariantCollisionObject>
    TimeVariantCollisionObjectConstPtr;

} // namespace collision

#endif
