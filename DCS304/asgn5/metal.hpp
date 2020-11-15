#pragma once

#include "material.hpp"

class metal : public material
{
public:
    metal(const vec3 &albedo, double f);
    virtual bool scatter(const ray &r_in, const hit_record &record, vec3 &attenuation, ray &scattered) const;

private:
    vec3 _albedo;
    double _f;
};
