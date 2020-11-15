#include "PathTracer.h"

#include <iostream>
#include <time.h>
#include "camera.hpp"
#include "hitable_list.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "tqdm.hpp"

constexpr size_t ANTIALIAS_ITER = 100;
constexpr size_t DEPTH_CUTOFF = 100;

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

vec3 ray_color(const ray &r, const hitable &world, size_t depth)
{
	hit_record record;
	if (world.hit(r, 0.0, std::numeric_limits<double>().max(), record))
	{
		vec3 target = record.p + record.norm + random_in_unit_sphere();
		if (depth > DEPTH_CUTOFF)
			return vec3(0.0, 0.0, 0.0);
		else
			return 0.5 * ray_color(ray(record.p, target - record.p), world, depth + 1);
	}
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

PathTracer::PathTracer()
	: m_channel(4), m_width(800), m_height(600), m_image(nullptr) {}

PathTracer::~PathTracer()
{
	if (m_image != nullptr)
		m_image;
	m_image = nullptr;
}

void PathTracer::initialize(int width, int height)
{
	m_width = width;
	m_height = height;
	if (m_image != nullptr)
		delete m_image;

	// allocate pixel buffer, RGBA format.
	m_image = new unsigned char[width * height * m_channel];
}

unsigned char * PathTracer::render(double & timeConsuming)
{
	if (m_image == nullptr)
	{
		std::cout << "Must call initialize() before rendering.\n";
		return nullptr;
	}

	// record start time.
	double startFrame = clock();

	hitable_list world;
	world.add(std::make_unique<sphere>(0.0, 0.0, -1.0, 0.5));
	world.add(std::make_unique<sphere>(0.0, -100.5, -1.0, 100));

	camera cam(m_width, m_height);

	tqdm bar;
	// render the image pixel by pixel.
	for (int y = m_height - 1; y >= 0; --y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			// std::cout << x << " : " << y << std::endl;
			// TODO: implement your ray tracing algorithm by yourself.
			vec3 color = vec3(0.0, 0.0, 0.0);
			for (size_t s = 0; s < ANTIALIAS_ITER; s++)
			{
				ray r = cam.get_ray_antialias(x, y);
				color += ray_color(r, world, 0);
			}
			color /= ANTIALIAS_ITER;
			color = vec3(std::sqrt(color.x()), std::sqrt(color.y()), std::sqrt(color.z()));

			drawPixel(x, y, color);
			bar.progress((m_height - y) * m_width + x, m_height * m_width);
		}
	}
	bar.finish();

	// record end time.
	double endFrame = clock();

	// calculate time consuming.
	timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;

	return m_image;
}

void PathTracer::drawPixel(unsigned int x, unsigned int y, const vec3 & color)
{
	// Check out 
	if (x < 0 || x >= m_width || y < 0 || y >= m_height)
		return;
	// x is column's index, y is row's index.
	unsigned int index = (y * m_width + x) * m_channel;
	// store the pixel.
	// red component.
	m_image[index + 0] = static_cast<unsigned char>(255 * color.x());
	// green component.
	m_image[index + 1] = static_cast<unsigned char>(255 * color.y());
	// blue component.
	m_image[index + 2] = static_cast<unsigned char>(255 * color.z());
	// alpha component.
	m_image[index + 3] = static_cast<unsigned char>(255);
}
