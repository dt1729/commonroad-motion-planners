#pragma once
#include "collision/serialize/collision_object_export_s11.h"
#include <s11n.net/s11n/s11nlite.hpp> // s11nlite framework

namespace collision {
namespace serialize {
class IShapeExport : public ICollisionObjectExport_s11 {};

} // namespace serialize

} // namespace collision
