#pragma once

#include "ray.hpp"
#include "hitable.hpp"

class material
{
public:
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const = 0;
};
