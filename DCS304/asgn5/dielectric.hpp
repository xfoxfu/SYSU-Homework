#pragma once

#include "material.hpp"

class dielectric : public material
{
public:
    dielectric(double ri);
    virtual bool scatter(const ray &r_in, const hit_record &record, vec3 &attenuation, ray &scattered) const;

private:
    double _ri;
};
