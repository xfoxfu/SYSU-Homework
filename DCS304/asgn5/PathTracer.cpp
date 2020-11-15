#include "PathTracer.h"

#include <iostream>
#include <time.h>
#include "camera.hpp"
#include "hitable_list.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "tqdm.hpp"
#include "material.hpp"
#include "lambertian.hpp"
#include "metal.hpp"
#include "dielectric.hpp"

constexpr size_t ANTIALIAS_ITER = 100;
constexpr size_t DEPTH_CUTOFF = 50;

vec3 ray_color(const ray &r, const hitable &world, size_t depth)
{
	hit_record record;
	if (world.hit(r, 0.0, std::numeric_limits<double>().max(), record))
	{
		ray scattered;
		vec3 attenuation;
		if (depth < DEPTH_CUTOFF && record.mat && record.mat->scatter(r, record, attenuation, scattered))
		{
			return attenuation * ray_color(scattered, world, depth + 1);
		}
		return vec3(0.0, 0.0, 0.0);
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
	world.add(std::make_unique<sphere>(0.0, 0.0, -1.0, 0.5, std::make_unique<lambertian>(vec3(0.1, 0.2, 0.5))));
	world.add(std::make_unique<sphere>(0.0, -100.5, -1.0, 100, std::make_unique<lambertian>(vec3(0.8, 0.8, 0.0))));
	world.add(std::make_unique<sphere>(1.0, 0, -1.0, 0.5, std::make_unique<metal>(vec3(0.8, 0.6, 0.2), 0.3)));
	world.add(std::make_unique<sphere>(-1.0, 0, -1.0, 0.5, std::make_unique<dielectric>(1.5)));
	world.add(std::make_unique<sphere>(-1.0, 0, -1.0, -0.45, std::make_unique<dielectric>(1.5)));

	vec3 lookfrom(3, 3, 2);
	vec3 lookat(0, 0, -1);
	camera cam(lookfrom, lookat, vec3(0, 1, 0), 20.0, m_width, m_height, 2.0, (lookfrom - lookat).length());

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
