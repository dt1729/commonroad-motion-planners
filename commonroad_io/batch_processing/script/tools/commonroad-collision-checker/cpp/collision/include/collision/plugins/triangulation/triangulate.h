#pragma once

#include "collision/application_settings.h"

#if ENABLE_TRIANGULATION
#include "collision/narrowphase/triangle.h"

namespace collision {
namespace triangulation {
int triangulate(std::vector<Eigen::Vector2d> vertices,
                std::vector<collision::TriangleConstPtr> &triangles_out);
int triangulateQuality(std::vector<Eigen::Vector2d> vertices,
                       std::vector<collision::TriangleConstPtr> &triangles_out,
                       double quality_b = 0.125);
} // namespace triangulation
} // namespace collision
#endif
