#pragma once

#include "hitable.hpp"
#include <vector>
#include <memory>

class hitable_list : public hitable
{
public:
    hitable_list();
    void add(std::unique_ptr<hitable> &&obj);
    virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

private:
    std::vector<std::unique_ptr<hitable>> _list;
};
