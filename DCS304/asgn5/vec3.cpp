#include "vec3.h"
#include <random>

vec3 random_in_unit_sphere()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    vec3 p;
    do
    {
        p = 2.0 * vec3(dist(gen), dist(gen), dist(gen)) - vec3(1.0, 1.0, 1.0);
    } while (p.squared_length() >= 1.0);
    return p;
}

vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0 * dot(v, n) * n;
}

bool refract(const vec3 &v, const vec3 &n, double ni_over_nt, vec3 &refracted)
{
    vec3 uv = unit_vector(v);
    double dt = dot(uv, n);
    double delta = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (delta > 0)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * std::sqrt(delta);
        return true;
    }
    return false;
}

vec3 random_in_unit_disk()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    vec3 p;
    do
    {
        p = 2.0 * vec3(dist(gen), dist(gen), 0.0) - vec3(1.0, 1.0, 0.0);
    } while (dot(p, p) >= 1.0);
    return p;
}
