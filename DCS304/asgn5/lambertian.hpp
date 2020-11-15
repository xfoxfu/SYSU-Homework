#pragma once

#include "material.hpp"

class lambertian : public material
{
public:
    lambertian(const vec3 &albedo);
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const;

private:
    vec3 _albedo;
};
