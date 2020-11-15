#include "dielectric.hpp"
#include <random>

double schlick(double cosine, double ri)
{
    double r0 = (1 - ri) / (1 + ri);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

dielectric::dielectric(double ri)
{
    _ri = ri;
}

bool dielectric::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const
{
    vec3 reflected = reflect(r_in.direction(), rec.norm);
    double ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 outward_normal;
    vec3 refracted;
    double reflect_prob;
    double cosine;
    if (dot(r_in.direction(), rec.norm) > 0)
    {
        outward_normal = -rec.norm;
        ni_over_nt = _ri;
        cosine = _ri * dot(r_in.direction(), rec.norm) / r_in.direction().length();
    }
    else
    {
        outward_normal = rec.norm;
        ni_over_nt = 1.0 / _ri;
        cosine = -dot(r_in.direction(), rec.norm) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
    {
        reflect_prob = schlick(cosine, _ri);
    }
    else
    {
        scattered = ray(rec.p, reflected);
        reflect_prob = 1.0;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double p = dist(gen);
    if (p < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scattered = ray(rec.p, refracted);

    return true;
}
