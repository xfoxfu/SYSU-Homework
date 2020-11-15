#pragma once

#include "ray.hpp"
#include <random>

class camera
{
public:
    camera(vec3 lookfrom, vec3 lookat, vec3 lookup, double vfov, size_t vw, size_t vh);
    ray get_ray(double x, double y) const;
    ray get_ray(size_t x, size_t y) const;
    ray get_ray(int x, int y) const;
    ray get_ray_antialias(size_t x, size_t y);
    ray get_ray_antialias(int x, int y);

private:
    vec3 _origin;
    vec3 _southwest;
    vec3 _horizontal;
    vec3 _vertical;
    size_t _vw;
    size_t _vh;
    std::random_device _rd;
    std::mt19937 _gen;
    std::uniform_real_distribution<double> _dis;
};
