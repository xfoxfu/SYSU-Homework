#include "ray.hpp"

ray::ray() {}

ray::ray(const vec3 &origin, const vec3 &direction)
{
    _origin = origin;
    _direction = direction;
}

const vec3 &ray::origin() const { return _origin; }

vec3 &ray::origin() { return _origin; }

const vec3 &ray::direction() const { return _direction; }

vec3 &ray::direction() { return _direction; }

vec3 ray::point_at_param(float t) const { return _origin + t * _direction; }
