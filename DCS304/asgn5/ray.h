#pragma once

#include "vec3.h"

class ray
{
public:
    ray();
    ray(const vec3 &origin, const vec3 &direction);
    const vec3 &origin() const;
    vec3 &origin();
    const vec3 &direction() const;
    vec3 &direction();
    vec3 point_at_param(float t) const;

private:
    vec3 _origin;
    vec3 _direction;
};
