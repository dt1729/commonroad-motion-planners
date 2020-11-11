#ifndef RECTANGLE_OBB_H_
#define RECTANGLE_OBB_H_

#include <exception>
#include <iostream>

#include "collision/narrowphase/shape.h"

#include "collision/line_segment.h"

namespace collision {
/*!
  \brief Oriented rectangle

*/
class RectangleOBB : public Shape {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  inline void set_up_segments(void) {
    Eigen::Vector2d _v1 =
        center() - r_x() * local_x_axis() - r_y() * local_y_axis();
    Eigen::Vector2d _v2 =
        center() + r_x() * local_x_axis() - r_y() * local_y_axis();
    Eigen::Vector2d _v3 =
        center() + r_x() * local_x_axis() + r_y() * local_y_axis();
    Eigen::Vector2d _v4 =
        center() - r_x() * local_x_axis() + r_y() * local_y_axis();

    segments_.push_back(LineSegment(_v1, _v2));
    segments_.push_back(LineSegment(_v2, _v3));
    segments_.push_back(LineSegment(_v3, _v4));
    segments_.push_back(LineSegment(_v4, _v1));
  }

  RectangleOBB(double _r_x, double _r_y, Eigen::Matrix2d _local_axes,
               const Eigen::Vector2d &_center = Eigen::Vector2d(0, 0))
      : Shape(_center), local_axes_(_local_axes), r_(_r_x, _r_y) {

    set_up_segments();
  }

  RectangleOBB(double _r_x, double _r_y, double angle,
               const Eigen::Vector2d &_center = Eigen::Vector2d(0, 0))
      : Shape(_center), r_(_r_x, _r_y) {
    local_axes_ << cos(angle), -sin(angle), sin(angle), cos(angle);

    set_up_segments();
  }

  virtual CollisionObjectType getCollisionObjectType() const {
    return CollisionObjectType::OBJ_TYPE_OBB_BOX;
  }

  bool rayTrace(const Eigen::Vector2d &point1, const Eigen::Vector2d &point2,
                std::vector<LineSegment> &intersect) const override;

  virtual ~RectangleOBB() {}

#if ENABLE_SERIALIZER
  serialize::ICollisionObjectExport *exportThis(void) const override;
#endif

  RectangleOBB(const RectangleOBB &copy);
  virtual RectangleOBB *clone() const;

  void print(std::ostringstream &stream) const;

  fcl::CollisionGeometry<FCL_PRECISION> *
  createFCLCollisionGeometry(void) const override;
  fcl::CollisionObject<FCL_PRECISION> *createFCLCollisionObject(
      const std::shared_ptr<fcl::CollisionGeometry<FCL_PRECISION>> &)
      const override;

  ShapeType type() const;

  Eigen::Matrix2d local_axes() const;
  Eigen::Vector2d local_x_axis() const;
  Eigen::Vector2d local_y_axis() const;
  Eigen::Vector2d r() const;
  double r(int i) const;
  double r_x() const;
  double r_y() const;

  void set_local_x_axis(const Eigen::Vector2d &x_axis);
  void set_local_y_axis(const Eigen::Vector2d &y_axis);
  void set_r_x(double _r_x);
  void set_r_y(double _r_y);

  //
  // @brief Computes the square distance between a point and
  //        the rectangle boundary. From:
  //        C. Ericson, Real-Time Collision Detection, pp. 134, 2004
  // @param p Point
  //
  double squareDisToPoint(const Eigen::Vector2d &p) const;

  std::vector<LineSegment> segments(void) const { return segments_; };

protected:
  using Shape::center_;
  using Shape::radius_;

  // Local x- and y-axis (column vectors)
  //               | x_1  x_2 |
  //  local_axis = |          |
  //               | y_1  y_2 |
  //
  Eigen::Matrix2d local_axes_;

  std::vector<LineSegment> segments_;

  //! Positive halfwidth extents of OBB along each axis
  Eigen::Vector2d r_;

  static constexpr ShapeType type_ = TYPE_RECTANGLE_OBB;
};

} // namespace collision

#endif
