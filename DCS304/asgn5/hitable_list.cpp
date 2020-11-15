#include "hitable_list.hpp"

hitable_list::hitable_list()
{
}

void hitable_list::add(std::unique_ptr<hitable> &&obj)
{
    _list.push_back(std::move(obj));
}

bool hitable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    double closest = t_max;
    bool hit = false;
    for (auto &obj : _list)
    {
        if (obj->hit(r, t_min, closest, rec))
        {
            hit = true;
            closest = rec.t;
        }
    }
    return hit;
}
