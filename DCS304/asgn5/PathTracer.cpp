#include "PathTracer.h"

#include "ray.h"
#include <iostream>
#include <time.h>

double hit_sphere_point(const vec3 &ce, double ra, const ray &ry)
{
	vec3 oc = ry.origin() - ce;
	double a = dot(ry.direction(), ry.direction());
	double b = 2.0 * dot(oc, ry.direction());
	double c = dot(oc, oc) - ra * ra;
	float delta = b * b - 4 * a * c;
	if (delta < 0)
		return -1.0;					   // no solution
	return (-b - sqrt(delta)) / (2.0 * a); // the nearer point
}

vec3 ray_color(const ray &r)
{
	float t = hit_sphere_point(vec3(0.0, 0.0, -1.0), 0.5, r);
	if (t > 0.0)
	{
		vec3 n = unit_vector(r.point_at_param(t) - vec3(0.0, 0.0, -1.0));
		return 0.5 * (n + vec3(1.0, 1.0, 1.0));
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

	double scale = static_cast<double>(m_width) / static_cast<double>(m_height);
	vec3 southwest(-scale, -1.0, -1.0);
	vec3 horizontal(scale * 2.0, 0.0, 0.0);
	vec3 vertical(0.0, 2.0, 0.0);
	vec3 origin(0.0, 0.0, 0.0);

	// render the image pixel by pixel.
	for (int y = m_height - 1; y >= 0; --y)
	{
		for (int x = 0; x < m_width; ++x)
		{
			// TODO: implement your ray tracing algorithm by yourself.
			double u = static_cast<double>(x) / static_cast<double>(m_width);
			double v = static_cast<double>(y) / static_cast<double>(m_height);
			ray r(origin, southwest + u * horizontal + v * vertical);
			vec3 color = ray_color(r);

			drawPixel(x, y, color);
		}
	}

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
