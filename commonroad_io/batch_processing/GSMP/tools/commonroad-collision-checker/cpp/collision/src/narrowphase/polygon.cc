#include "collision/plugins/triangulation/triangulate.h"
#include "collision/raytrace_utils.h"
#include "collision/solvers/fcl/fcl_decl.h"
#include "collision/solvers/fcl/fcl_transform.h"
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/collision_object.h>

#include "collision/narrowphase/polygon.h"

namespace collision {

Polygon::Polygon(std::vector<Eigen::Vector2d> &vertices,
                 std::vector<std::vector<Eigen::Vector2d>> &hole_vertices,
                 std::vector<TriangleConstPtr> &mesh_triangles,
                 const Eigen::Vector2d &_center)
    : Shape(_center) {
  vertices_ = vertices;
  hole_vertices_ = hole_vertices;
  mesh_triangles_ = mesh_triangles;
  invalidateCollisionEntityCache();
}
#if ENABLE_TRIANGULATION
Polygon::Polygon(std::vector<Eigen::Vector2d> &vertices, double mesh_quality_b,
                 const Eigen::Vector2d &_center)
    : Shape(_center) {
  vertices_ = vertices;
  if (!mesh_quality_b) {
    triangulation::triangulate(vertices, mesh_triangles_);
  } else {
    triangulation::triangulateQuality(vertices, mesh_triangles_,
                                      mesh_quality_b);
  }
  invalidateCollisionEntityCache();
}
#endif

bool Polygon::rayTrace(const Eigen::Vector2d &point1,
                       const Eigen::Vector2d &point2,
                       std::vector<LineSegment> &intersect) const {

  LineSegment obj_segment(point1, point2);
  bool res = false;
  for (auto &tri : mesh_triangles_) {
    std::vector<Eigen::Vector2d> inters1;
    for (auto &segm : tri->segments()) {
      res = segm.intersect(obj_segment, inters1) || res;
    }
    collision::raytrace::rayTracePostprocess(point1, point2, inters1, intersect,
                                             tri.get());
  }
  return res;
}

Polygon *Polygon::clone() const { return new Polygon(*this); }

Polygon::Polygon(const Polygon &copy) : Shape(copy) {
  center_ = copy.center(); // not needed? It is already set in Shape(copy)
  radius_ = copy.radius();
  vertices_ = copy.getVertices();
  hole_vertices_ = copy.getHoleVertices();
  mesh_triangles_ = copy.getTriangleMesh();
  invalidateCollisionEntityCache();
}

ShapeType Polygon::type(void) const { return type_; }

void Polygon::print(std::ostringstream &stream) const {}

void Polygon::toString(std::ostringstream &stream) const {
  stream << "Polygon "
         << "triangles: ";
  for (auto &tri : mesh_triangles_) {
    tri->print(stream);
  }
  stream << "\\Polygon " << std::endl;
}

fcl::CollisionGeometry<FCL_PRECISION> *
Polygon::createFCLCollisionGeometry(void) const {
  fcl::BVHModel<fcl::AABB<FCL_PRECISION>> *model =
      new fcl::BVHModel<fcl::AABB<FCL_PRECISION>>();
  model->beginModel(mesh_triangles_.size(), mesh_triangles_.size() * 3);
  fcl::Vector3<FCL_PRECISION> v3(0, 0, 0);
  for (auto &tr : mesh_triangles_) {
    Eigen::Vector2d v2d = tr->v1();
    fcl::Vector3<FCL_PRECISION> v1(v2d[0], v2d[1], 0);
    v2d = tr->v2();
    fcl::Vector3<FCL_PRECISION> v2(v2d[0], v2d[1], 0);
    v2d = tr->v3();
    fcl::Vector3<FCL_PRECISION> v3(v2d[0], v2d[1], 0);
    model->addTriangle(v1, v2, v3);
  }
  model->endModel();
  return model;
}
fcl::CollisionObject<FCL_PRECISION> *Polygon::createFCLCollisionObject(
    const std::shared_ptr<fcl::CollisionGeometry<FCL_PRECISION>> &col_geom)
    const {

  return new fcl::CollisionObject<FCL_PRECISION>(col_geom);
}

CollisionObjectConstPtr
Polygon::timeSlice(int time_idx,
                   CollisionObjectConstPtr shared_ptr_this) const {
  return shared_ptr_this;
}

std::vector<TriangleConstPtr> Polygon::getTriangleMesh() const {
  return mesh_triangles_;
}

std::vector<Eigen::Vector2d> Polygon::getVertices() const { return vertices_; }

std::vector<std::vector<Eigen::Vector2d>> Polygon::getHoleVertices() const {
  return hole_vertices_;
}

#if ENABLE_SERIALIZER

namespace serialize {
ICollisionObjectExport *exportObject(const collision::Polygon &);
}

serialize::ICollisionObjectExport *Polygon::exportThis(void) const {
  return serialize::exportObject(*this);
}
#endif

} // namespace collision
