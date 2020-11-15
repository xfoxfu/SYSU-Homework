#include "vec3.h"
#include <random>

vec3 random_in_unit_sphere()
{
    vec3 p;
    do
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        p = 2.0 * vec3(dist(gen), dist(gen), dist(gen)) - vec3(1.0, 1.0, 1.0);
    } while (p.squared_length() >= 1.0);
    return p;
}
