#include "camera.hpp"

camera::camera(vec3 lookfrom, vec3 lookat, vec3 lookup, double vfov, size_t vw, size_t vh)
{
    vec3 u, v, w;
    _vw = vw;
    _vh = vh;
    double aspect = static_cast<double>(vw) / static_cast<double>(vh);

    double theta = vfov * M_PI / 180.0;
    double half_h = tan(theta / 2);
    double half_w = aspect * half_h;

    _origin = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(lookup, w));
    v = cross(w, u);
    _southwest = _origin - half_w * u - half_h * v - w;
    _horizontal = 2 * half_w * u;
    _vertical = 2 * half_h * v;

    _gen = std::mt19937(_rd());
    _dis = std::uniform_real_distribution<double>(0.0, 1.0);
}

ray camera::get_ray(double u, double v) const
{
    return ray(_origin, _southwest + u * _horizontal + v * _vertical - _origin);
}

ray camera::get_ray(size_t x, size_t y) const
{
    return get_ray(static_cast<double>(x) / static_cast<double>(_vw),
                   static_cast<double>(y) / static_cast<double>(_vh));
}

ray camera::get_ray_antialias(size_t x, size_t y)
{
    return get_ray((static_cast<double>(x) + _dis(_gen)) / static_cast<double>(_vw),
                   (static_cast<double>(y) + _dis(_gen)) / static_cast<double>(_vh));
}

ray camera::get_ray(int x, int y) const
{
    return get_ray(static_cast<size_t>(x), static_cast<size_t>(y));
}

ray camera::get_ray_antialias(int x, int y)
{
    return get_ray_antialias(static_cast<size_t>(x), static_cast<size_t>(y));
}
