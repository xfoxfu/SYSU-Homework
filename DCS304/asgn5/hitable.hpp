#pragma once

#include "vec3.h"
#include "ray.h"

struct hit_record
{
    double t;
    vec3 p;
    vec3 norm;
};

class hitable
{
public:
    virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
};
