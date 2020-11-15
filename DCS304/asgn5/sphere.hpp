#pragma once

#include "hitable.hpp"
#include "material.hpp"

class sphere : public hitable
{
public:
    sphere();
    sphere(vec3 center, double radius, std::unique_ptr<material> &&material);
    sphere(double x, double y, double z, double radius, std::unique_ptr<material> &&material);
    virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    vec3 center;
    double radius;
    std::unique_ptr<material> material;
};
