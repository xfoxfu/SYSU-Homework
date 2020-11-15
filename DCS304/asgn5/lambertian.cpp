#include "lambertian.hpp"
#include <random>

lambertian::lambertian(const vec3 &albedo)
{
    _albedo = albedo;
}

bool lambertian::scatter(const ray &r_in, const hit_record &record, vec3 &attenuation, ray &scattered) const
{
    vec3 target = record.p + record.norm + random_in_unit_sphere();
    scattered = ray(record.p, target - record.p);
    attenuation = _albedo;
    return true;
}
