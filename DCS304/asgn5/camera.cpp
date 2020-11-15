#include "camera.hpp"

camera::camera(vec3 lookfrom, vec3 lookat, vec3 lookup,
               double vfov, size_t vw, size_t vh,
               double aperture, double focus_dist)
{
    _lens_radius = aperture / 2;
    _vw = vw;
    _vh = vh;
    double aspect = static_cast<double>(vw) / static_cast<double>(vh);

    double theta = vfov * M_PI / 180.0;
    double half_h = tan(theta / 2) * focus_dist;
    double half_w = aspect * half_h; // already timed focus_dist

    _origin = lookfrom;
    _w = unit_vector(lookfrom - lookat);
    _u = unit_vector(cross(lookup, _w));
    _v = cross(_w, _u);
    _southwest = _origin - half_w * _u - half_h * _v - _w * focus_dist;
    _horizontal = 2 * half_w * _u;
    _vertical = 2 * half_h * _v;

    _gen = std::mt19937(_rd());
    _dis = std::uniform_real_distribution<double>(0.0, 1.0);
}

ray camera::get_ray(double u, double v) const
{
    vec3 rd = _lens_radius * random_in_unit_disk();
    vec3 offset = _u * rd.x() + _v * rd.y();
    return ray(_origin + offset, _southwest + u * _horizontal + v * _vertical - _origin - offset);
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
