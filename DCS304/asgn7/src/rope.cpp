#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

constexpr double damping_factor = 0.01;

namespace CGL {

    Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass, float k, vector<int> pinned_nodes)
    {
        auto delta = (end - start) / num_nodes;
        for (size_t i = 0; i < num_nodes; i++)
        {
            masses.push_back(new Mass(start + i * delta, node_mass, false));
            if (i > 0)
            {
                springs.push_back(new Spring(masses[i - 1], masses[i], k));
            }
        }
        // Comment-in this part when you implement the constructor
        for (auto &i : pinned_nodes)
        {
            masses[i]->pinned = true;
        }
    }

    void Rope::simulateEuler(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 2): Use Hooke's law to calculate the force on a node
            auto delta = s->m2->position - s->m1->position;
            s->m1->forces += s->k * delta.unit() * (delta.norm() - s->rest_length);
            s->m2->forces += -s->k * delta.unit() * (delta.norm() - s->rest_length);
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                // TODO (Part 2): Add gravity and global damping, then compute the new velocity and position
                auto acceleration = (gravity + m->forces - damping_factor * m->velocity) / m->mass;
                // m->position += m->velocity * delta_t; // explicit
                m->velocity += acceleration * delta_t;
                m->position += m->velocity * delta_t; // semi-implicit
            }

            // Reset all forces on each mass
            m->forces = Vector2D(0, 0);
        }
    }

    void Rope::simulateVerlet(float delta_t, Vector2D gravity)
    {
        for (auto &s : springs)
        {
            // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet （solving constraints)
            // 在此进行质点的位置调整，维持弹簧的原始长度
            auto delta = s->m2->position - s->m1->position;
            if (!s->m1->pinned)
                s->m1->position += (delta.norm() - s->rest_length) / 2 * delta.unit();
            if (!s->m2->pinned)
                s->m2->position += -(delta.norm() - s->rest_length) / 2 * delta.unit();
        }

        for (auto &m : masses)
        {
            if (!m->pinned)
            {
                Vector2D temp_position = m->position;
                // TODO (Part 3): Set the new position of the rope mass
                // 在此计算重力影响下，质点的位置变化
                m->position += (1 - damping_factor) * (m->position - m->last_position) + gravity * delta_t / 2 * delta_t / 2;
                m->last_position = temp_position;
            }
        }
    }
}
