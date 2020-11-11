#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <Eigen/Dense>

#include "collision/narrowphase/rectangle_aabb.h"
#include "collision/narrowphase/rectangle_obb.h"
#include "collision/collision_checker.h"
#include "collision/shape_group.h"
#include "collision/narrowphase/polygon.h"
#include "collision/narrowphase/sphere.h"
#include "collision/narrowphase/triangle.h"
#include "collision/narrowphase/point.h"
#include "collision/time_variant_collision_object.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

void init_module_collision(py::module &m) {

  py::class_<collision::CollisionObject,
             std::shared_ptr<collision::CollisionObject> >(m, "CollisionObject");

  py::class_<collision::Shape,
             collision::CollisionObject, std::shared_ptr<collision::Shape> >(m, "Shape");

  py::class_<collision::Point, collision::Shape,
             std::shared_ptr<collision::Point> >(m, "Point")
      .def(py::init([](double x, double y) {
        return new collision::Point(Eigen::Vector2d(x, y));
      }), py::arg("x"), py::arg("y"))
      .def("collide", [](std::shared_ptr<collision::Point> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("center", [](collision::Point &point) {
        Eigen::Vector2d tmp = point.center();
        std::vector<double> pos;
        pos.push_back(tmp(0));
        pos.push_back(tmp(1));
        return py::array(2, pos.data());
      });

  py::class_<collision::RectangleAABB, collision::Shape,
             std::shared_ptr<collision::RectangleAABB> >(m, "RectAABB")
      .def(py::init([](double r_x, double r_y, double x, double y) {
        return new collision::RectangleAABB(r_x, r_y, Eigen::Vector2d(x, y));
      }), py::arg("width/2"), py::arg("height/2"), py::arg("center x"), py::arg("center y"))
      .def("collide", [](std::shared_ptr<collision::RectangleAABB> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("__str__", [](const std::shared_ptr<collision::RectangleAABB> &c) {
        return "<collision::RectangleAABB r_x=" + std::to_string(c->r_x()) +
            " r_y=" + std::to_string(c->r_y()) + ">";
      })
      .def("set_all", &collision::RectangleAABB::set_all)
      .def("min_x", [](const std::shared_ptr<collision::RectangleAABB> &c) {
        Eigen::Vector2d tmp = c->min();
        return tmp(0);
      })
      .def("min_y", [](const std::shared_ptr<collision::RectangleAABB> &c) {
        Eigen::Vector2d tmp = c->min();
        return tmp(1);
      })
      .def("max_x", [](const std::shared_ptr<collision::RectangleAABB> &c) {
        Eigen::Vector2d tmp = c->max();
        return tmp(0);
      })
      .def("max_y", [](const std::shared_ptr<collision::RectangleAABB> &c) {
        Eigen::Vector2d tmp = c->max();
        return tmp(1);
      })
      .def_property_readonly("r_x", &collision::RectangleAABB::r_x);

  py::class_<collision::RectangleOBB, collision::Shape,
             std::shared_ptr<collision::RectangleOBB> >(
      m, "RectOBB")
      .def(py::init([](double r_x, double r_y, double angle, double x, double y) {
        return new collision::RectangleOBB(r_x, r_y, angle, Eigen::Vector2d(x, y));
      }), py::arg("width/2"), py::arg("height/2"), py::arg("orientation"), py::arg("center x"), py::arg("center y"))
      .def("collide", [](std::shared_ptr<collision::RectangleOBB> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("set_center", [](collision::RectangleOBB &rect, double x, double y) {
        rect.set_center(Eigen::Vector2d(x, y));
      })
      .def("local_x_axis", [](collision::RectangleOBB &rect) {
        Eigen::Vector2d tmp = rect.local_x_axis();
        std::vector<double> pos;
        pos.push_back(tmp(0));
        pos.push_back(tmp(1));
        return py::array(2, pos.data());
      })
      .def("local_y_axis", [](collision::RectangleOBB &rect) {
        Eigen::Vector2d tmp = rect.local_y_axis();
        std::vector<double> pos;
        pos.push_back(tmp(0));
        pos.push_back(tmp(1));
        return py::array(2, pos.data());
      })
      .def("r_x", &collision::RectangleOBB::r_x, "Positive halfwidth extent of OBB along local x-axis")
      .def("r_y", &collision::RectangleOBB::r_y, "Positive halfwidth extent of OBB along local y-axis")
      .def("center", [](collision::RectangleOBB &rect) {
        Eigen::Vector2d tmp = rect.center();
        std::vector<double> pos;
        pos.push_back(tmp(0));
        pos.push_back(tmp(1));
        return py::array(2, pos.data());
      })
      .def("__str__", [](collision::RectangleOBB &c) {
        return "<collision::RectangleOBB r_x=" + std::to_string(c.r_x()) +
            " r_y=" + std::to_string(c.r_y()) + " center_x="
            + std::to_string(c.center_x()) + " center_y="
            + std::to_string(c.center_y()) + ">";
      });

  py::class_<collision::Triangle, collision::Shape,
             std::shared_ptr<collision::Triangle> >(m, "Triangle")
      .def(py::init([](double x1, double y1, double x2, double y2, double x3, double y3) {
        return new collision::Triangle(Eigen::Vector2d(x1, y1),
                                       Eigen::Vector2d(x2, y2),
                                       Eigen::Vector2d(x3, y3));
      }), py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("x3"), py::arg("y3"))
      .def("collide", [](std::shared_ptr<collision::Triangle> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("vertices", [](collision::Triangle &obj) {
        py::list v1;
        v1.append(py::cast(obj.v1()[0]));
        v1.append(py::cast(obj.v1()[1]));
        py::list v2;
        v2.append(py::cast(obj.v2()[0]));
        v2.append(py::cast(obj.v2()[1]));
        py::list v3;
        v3.append(py::cast(obj.v3()[0]));
        v3.append(py::cast(obj.v3()[1]));

        py::list ret_list;
        ret_list.append(v1);
        ret_list.append(v2);
        ret_list.append(v3);
        return ret_list;
      })
      .def("__str__", [](collision::Triangle &c) {
        return "<collision::Triangle v1=" + std::to_string(c.v1()[0]) + "/"
            + std::to_string(c.v1()[1]) + " v2=" + std::to_string(c.v2()[0])
            + "/" + std::to_string(c.v2()[1]) + " v3="
            + std::to_string(c.v3()[0]) + "/" + std::to_string(c.v3()[1])
            + ">";
      });

  py::class_<collision::Sphere, collision::Shape,
             std::shared_ptr<collision::Sphere> >(m, "Circle")
      .def(py::init<double, double, double>(), py::arg("radius"), py::arg("center x"), py::arg("center y"))
      .def("collide", [](std::shared_ptr<collision::Sphere> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("r", &collision::Sphere::radius, "radius of the circle")
      .def("center", [](collision::Sphere &sphere) {
        Eigen::Vector2d tmp = sphere.center();
        std::vector<double> pos;
        pos.push_back(tmp(0));
        pos.push_back(tmp(1));
        return py::array(2, pos.data());
      })
      .def("x", &collision::Sphere::get_x, "x-coordinate of center")
      .def("y", &collision::Sphere::get_y, "y-coordinate of center");

  py::class_<collision::TimeVariantCollisionObject, collision::CollisionObject,
             std::shared_ptr<collision::TimeVariantCollisionObject> >(m, "TimeVariantCollisionObject")
      .def(py::init<int>(), py::arg("time_start_idx"))
      .def("collide", [](
          std::shared_ptr<collision::TimeVariantCollisionObject> &cc,
          std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("append_obstacle", [](collision::TimeVariantCollisionObject &obj,
                                 std::shared_ptr<collision::CollisionObject> co) {
        obj.appendObstacle(co);
      })
      .def("time_start_idx", &collision::TimeVariantCollisionObject::time_start_idx)
      .def("time_end_idx", &collision::TimeVariantCollisionObject::time_end_idx)
      .def("obstacle_at_time",
           &collision::TimeVariantCollisionObject::getObstacleAtTime);

  py::class_<collision::ShapeGroup, collision::CollisionObject,
             std::shared_ptr<collision::ShapeGroup> >(m, "ShapeGroup")
      .def(py::init<>())
      .def("collide", [](std::shared_ptr<collision::ShapeGroup> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("overlap", [](std::shared_ptr<collision::ShapeGroup> &cc,
                         std::shared_ptr<collision::ShapeGroup> &co) {
        std::vector<std::pair<int, int> > idx = cc->overlap(*co);
        py::list ret;
        for (auto &i : idx) {
          ret.append(py::cast(i));
        }
        return ret;
      })
      .def("overlap_map", [](std::shared_ptr<collision::ShapeGroup> &cc,
                             std::shared_ptr<collision::ShapeGroup> &co) {
        std::vector<std::set<int> > idx = cc->overlapMap(*co);
        py::dict ret;
        int key = 0;
        for (auto &i : idx) {
          py::set app1;
          for (auto &j : i) {
            app1.add(py::cast(j));
          }
          ret[py::cast(key)] = app1;
          key++;
        }
        return ret;
      })
      .def("add_shape", [](collision::ShapeGroup &obj,
                           std::shared_ptr<collision::Shape> co) {
        obj.addToGroup(co);
      })
      .def("size", [](collision::ShapeGroup &obj) {
        auto unpacked = obj.unpack();
        return unpacked.size();
      })
      .def("unpack", [](collision::ShapeGroup &obj) {
        auto unpacked = obj.unpack();
        py::list ret_list;
        for (auto &i : unpacked) {
          ret_list.append(py::cast(i));
        }
        return ret_list;
      });

  py::class_<collision::Polygon, collision::Shape,
             std::shared_ptr<collision::Polygon> >(m, "Polygon")
      .def(py::init([](std::vector<std::array<double, 2> > outer_boundary,
                       py::list holes, py::list python_mesh_triangles) {
        std::vector<Eigen::Vector2d> vertices;
        std::vector<std::vector<Eigen::Vector2d> > hole_vertices;
        std::vector<collision::TriangleConstPtr> mesh_triangles;

        for (const auto &vertex : outer_boundary) {
          vertices.push_back(Eigen::Vector2d(vertex[0], vertex[1]));
        }

        for (const auto &triangle : python_mesh_triangles) {
          mesh_triangles.push_back(triangle.cast<collision::TriangleConstPtr>());
        }
        return new collision::Polygon(vertices, hole_vertices,
                                      mesh_triangles);
      }), py::arg("outer_boundary"), py::arg("holes"), py::arg("triangle mesh"))
#if ENABLE_TRIANGULATION
      .def(py::init([](std::vector<std::array<double, 2> > outer_boundary, double mesh_quality) {
        std::vector<Eigen::Vector2d> vertices;
        for (const auto &vertex : outer_boundary) {
          vertices.push_back(Eigen::Vector2d(vertex[0], vertex[1]));
        }
        return new collision::Polygon(vertices, mesh_quality);
      }))
#endif
      .def("collide", [](std::shared_ptr<collision::Polygon> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(*co);
      })
      .def("triangle_mesh", [](collision::Polygon &obj) {
        auto triangles = obj.getTriangleMesh();
        py::list ret_list;
        for (collision::TriangleConstPtr &i : triangles) {
          ret_list.append(i);
        }
        return ret_list;
      })
      .def("vertices", [](collision::Polygon &obj) {
        py::list vertices;
        for (const auto &vertex : obj.getVertices()) {
          py::list point;
          point.append(vertex(0));
          point.append(vertex(1));
          vertices.append(point);
        }
        return vertices;
      })
      .def("__str__", [](collision::Polygon &c) {
        std::stringstream ss;
        ss << "<collision::Polygon vertices=";
        for (const auto &v : c.getVertices()) {
          ss << "(" << v(0) << "/" << v(1) << ") ";
        }
        ss << "\n";
        ss << ">";
        std::string s = ss.str();
        return ss.str();
      });

  py::class_<collision::CollisionChecker,
             std::shared_ptr<collision::CollisionChecker> >(m, "CollisionChecker")
      .def(py::init<>())
      .def("number_of_obstacles", &collision::CollisionChecker::numberOfObstacles)
      .def("__str__", [](const std::shared_ptr<collision::CollisionChecker> &c) {
        std::ostringstream stream;
        c->print(stream);
        return "<collision::CollisionChecker\n" + stream.str() + ">";
      })
      .def("add_collision_object",
           &collision::CollisionChecker::addCollisionObject)
      .def("collide", [](const std::shared_ptr<collision::CollisionChecker> &cc,
                         std::shared_ptr<collision::CollisionObject> &co) {
        return cc->collide(co);
      })
      .def("raytrace", [](
               const std::shared_ptr<collision::CollisionChecker> &cc,
               double point1_x, double point1_y, double point2_x, double point2_y, bool join_intervals) {
             std::vector<collision::LineSegment> res_seg;
             bool intersects = cc->rayTrace(Eigen::Vector2d(point1_x, point1_y),
                                            Eigen::Vector2d(point2_x, point2_y),
                                            res_seg, join_intervals);
             py::list ret_list;
             for (auto &i : res_seg) {
               py::list ret_sublist;
               ret_sublist.append(py::cast(i.point1().x));
               ret_sublist.append(py::cast(i.point1().y));
               ret_sublist.append(py::cast(i.point2().x));
               ret_sublist.append(py::cast(i.point2().y));

               ret_list.append(ret_sublist);
             }

             return ret_list;
           }
      )
      .def("find_all_colliding_objects", [](
          const std::shared_ptr<collision::CollisionChecker> &cc,
          std::shared_ptr<collision::CollisionObject> &co) {
        std::vector<collision::CollisionObjectConstPtr> obstacles;
        bool collides = cc->collide(co, obstacles);
        py::list ret_list;
        for (auto &i : obstacles) {
          ret_list.append(py::cast(i));
        }
        return ret_list;
      })
      .def("time_slice", &collision::CollisionChecker::timeSlice)
      .def("window_query", &collision::CollisionChecker::windowQuery)
      .def("obstacles", [](
          const std::shared_ptr<collision::CollisionChecker> &cc) {
        auto obstacles = cc->getObstacles();
        py::list ret_list;
        for (auto &o : obstacles) {
          ret_list.append(py::cast(o));
        }
        return ret_list;
      })
      .def("any_collides", [](
          const std::shared_ptr<collision::CollisionChecker> &cc,
          py::list py_obstacle_list) {
        std::vector<collision::CollisionObjectConstPtr> collision_object_list;
        for (const auto &item : py_obstacle_list) {
          if (cc->collide(item.cast<collision::CollisionObjectConstPtr>())) {
            return true;
          }
        }
        return false;
      });

}
