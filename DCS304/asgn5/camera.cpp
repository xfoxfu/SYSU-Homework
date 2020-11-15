#include "camera.hpp"

camera::camera(size_t vw, size_t vh)
{
    _vw = vw;
    _vh = vh;
    double scale = static_cast<double>(vw) / static_cast<double>(vh);
    _southwest = vec3(-scale, -1.0, -1.0);
    _horizontal = vec3(scale * 2.0, 0.0, 0.0);
    _vertical = vec3(0.0, 2.0, 0.0);
    _origin = vec3(0.0, 0.0, 0.0);

    _gen = std::mt19937(_rd());
    _dis = std::uniform_real_distribution<double>(0.0, 1.0);
}

ray camera::get_ray(double u, double v) const
{
    return ray(_origin, _southwest + u * _horizontal + v * _vertical);
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
