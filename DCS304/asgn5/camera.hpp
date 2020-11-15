#pragma once

#include "ray.hpp"

class camera
{
public:
    camera(size_t vw, size_t vh);
    camera(size_t vw, size_t vh, vec3 origin);
    ray get_ray(double x, double y) const;
    ray get_ray(size_t x, size_t y) const;
    ray get_ray(int x, int y) const;

private:
    vec3 _origin;
    vec3 _southwest;
    vec3 _horizontal;
    vec3 _vertical;
    size_t _vw;
    size_t _vh;
};
