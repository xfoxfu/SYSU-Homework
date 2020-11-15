#pragma once

#include "hitable.hpp"

class sphere : public hitable
{
public:
    sphere();
    sphere(vec3 center, double radius);
    sphere(double x, double y, double z, double radius);
    virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    vec3 center;
    double radius;
};
