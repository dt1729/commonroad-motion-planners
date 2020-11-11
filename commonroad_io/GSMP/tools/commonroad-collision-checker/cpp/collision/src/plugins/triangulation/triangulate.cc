#include "collision/plugins/triangulation/triangulate.h"

#define ENABLE_CGAL 0
#if ENABLE_TRIANGULATION
#if ENABLE_CGAL

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>

namespace collision {
namespace triangulation {
int triangulate(std::vector<Eigen::Vector2d> vertices,
                std::vector<collision::TriangleConstPtr> &triangles_out) {
  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Exact_predicates_tag Itag;
  typedef CGAL::Constrained_Delaunay_triangulation_2<K, CGAL::Default, Itag>
      CDT;
  typedef CDT::Point Point;
  typedef CGAL::Polygon_2<K> Polygon_2;

  CDT cdt;

  Polygon_2 polygon1;
  for (auto &el : vertices) {
    polygon1.push_back(Point(el[0], el[1]));
  }
  cdt.insert_constraint(polygon1.vertices_begin(), polygon1.vertices_end(),
                        true);

  for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
       fit != cdt.finite_faces_end(); ++fit) {

    Eigen::Vector2d vert1(cdt.triangle(fit)[0].x(), cdt.triangle(fit)[0].y());
    Eigen::Vector2d vert2(cdt.triangle(fit)[1].x(), cdt.triangle(fit)[1].y());
    Eigen::Vector2d vert3(cdt.triangle(fit)[2].x(), cdt.triangle(fit)[2].y());
    triangles_out.emplace_back(new collision::Triangle(vert1, vert2, vert3));
  }
  return 0;
}

int triangulateQuality(std::vector<Eigen::Vector2d> vertices,
                       std::vector<collision::TriangleConstPtr> &triangles_out,
                       double quality_b) {

  typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
  typedef CGAL::Triangulation_vertex_base_2<K> Vb;
  typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
  typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
  typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
  typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
  typedef CDT::Vertex_handle Vertex_handle;
  typedef CDT::Point Point;
  typedef CGAL::Polygon_2<K> Polygon_2;

  Polygon_2 polygon1;
  for (auto &el : vertices) {
    polygon1.push_back(Point(el[0], el[1]));
  }
  CDT cdt;
  cdt.insert_constraint(polygon1.vertices_begin(), polygon1.vertices_end(),
                        true);

  CGAL::refine_Delaunay_mesh_2(cdt, Criteria(quality_b, 0));

  for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
       fit != cdt.finite_faces_end(); ++fit) {

    Eigen::Vector2d vert1(cdt.triangle(fit)[0].x(), cdt.triangle(fit)[0].y());
    Eigen::Vector2d vert2(cdt.triangle(fit)[1].x(), cdt.triangle(fit)[1].y());
    Eigen::Vector2d vert3(cdt.triangle(fit)[2].x(), cdt.triangle(fit)[2].y());
    triangles_out.emplace_back(new collision::Triangle(vert1, vert2, vert3));
  }

  return 0;
}

} // namespace triangulation
} // namespace collision
#endif
#endif
