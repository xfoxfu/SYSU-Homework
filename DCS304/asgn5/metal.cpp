#include "metal.hpp"
#include <random>

vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0 * dot(v, n) * n;
}

metal::metal(const vec3 &albedo, double f)
{
    _albedo = albedo;
    _f = std::min(f, 1.0);
}

bool metal::scatter(const ray &r_in, const hit_record &record, vec3 &attenuation, ray &scattered) const
{
    vec3 reflected = reflect(unit_vector(r_in.direction()), record.norm);
    scattered = ray(record.p, reflected + _f * random_in_unit_sphere());
    attenuation = _albedo;
    return true;
}
