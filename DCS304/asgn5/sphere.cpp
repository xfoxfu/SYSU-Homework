#include "sphere.hpp"

sphere::sphere()
{
}

sphere::sphere(vec3 center, double radius,
               std::unique_ptr<::material> &&material) : center(center), radius(radius), material(move(material))
{
}

sphere::sphere(double x, double y, double z, double radius,
               std::unique_ptr<::material> &&material) : sphere(vec3(x, y, z), radius, move(material))
{
}

bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double b = 2.0 * dot(oc, r.direction());
    double c = dot(oc, oc) - radius * radius;
    float delta = b * b - 4 * a * c;

    if (delta < 0)
        return false; // no hit point

    double hitpoint = (-b - sqrt(delta)) / (2.0 * a);
    if (hitpoint > t_min && hitpoint < t_max) // the near point is valid
    {
        rec.t = hitpoint;
        rec.p = r.point_at_param(hitpoint);
        rec.norm = (rec.p - center) / radius;
        rec.mat = &*material;
        return true;
    }

    hitpoint = (-b + sqrt(delta)) / (2.0 * a);
    if (hitpoint > t_min && hitpoint < t_max) // the far point is valid
    {
        rec.t = hitpoint;
        rec.p = r.point_at_param(hitpoint);
        rec.norm = (rec.p - center) / radius;
        rec.mat = &*material;
        return true;
    }

    return false;
}
