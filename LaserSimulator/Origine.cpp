/*
* computePointCloud.cpp
* Created on: 10/12/2015
* Last Update: 21/12/2015
* Author: Nicola Rigato 1110346 
*
*/

#define _CRT_SECURE_NO_DEPRECATE
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/ros/conversions.h>
#include <pcl/conversions.h>
#include <pcl/octree/octree.h>
#include <pcl/common/angles.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <CL\cl2.hpp>
#include <thread>

using namespace cv;
using namespace std;
using namespace pcl;
using boost::chrono::high_resolution_clock;
using boost::chrono::duration;

#define EPSILON 0.000001
#define PIXEL_DIMENSION 0.0055 // mm

#define DIRECTION_SCAN_AXIS_X 0
#define DIRECTION_SCAN_AXIS_Y 1

#define LASER_1 -1
#define LASER_2 1

#define X 0
#define Y 1
#define Z 2

Eigen::Matrix<double, 3, 1> typedef Vect3d;

// MIN MAX POINT
float min_x, max_x;
float min_y, max_y;
float min_z, max_z;

PointXYZRGB laser_point, laser_point_2, laser_final_point_left, laser_final_point_right, pin_hole;

int scanDirection = DIRECTION_SCAN_AXIS_X;
float distance_laser_sensor = 600 ; // [500, 800]
float laser_aperture = 45.0;		// [30, 45]
float laser_inclination = 60.0;		// [60, 70]
float delta_z = 800;				// 600 altezza rispetto all'oggetto
float RAY_DENSITY = 0.001;
int default_number_samples = 10000000;

float camera_fps = 100;				// fps  [100, 500]
float scan_speed = 100;				// mm/s [100, 1000]
int sensor_pixel_width = 2024;
int sensor_pixel_height = 1088;
float focal_distance = 25;

float sensor_width = sensor_pixel_width * PIXEL_DIMENSION;
float sensor_height = sensor_pixel_height * PIXEL_DIMENSION;

float *min_poligon_point;
int   *min_poligon_index;

float DIRECTION_TAN_LASER_INCLINATION = tan(deg2rad(90 - laser_inclination));
float DIRECTION_TAN_LASER_APERTURE = tan(deg2rad(laser_aperture / 2));

struct Plane{
	float A;
	float B;
	float C;
	float D;
};

struct Vec3
{
	float points[3];
};
struct Triangle {
	Vec3 vertex1;
	Vec3 vertex2;
	Vec3 vertex3;
};

struct Triangle3f {
	cl_float3 vertex1;
	cl_float3 vertex2;
	cl_float3 vertex3;
};


struct OpenCLDATA {
	cl::Buffer device_triangle_array;
	cl::Buffer device_output_points;
	cl::Buffer device_output_hits;

	size_t triangles_size;
	size_t points_size;
	size_t hits_size;

	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
	cl::Program program_;

	std::vector<cl::Device> devices;
	std::vector<cl::Platform> platforms;
};

void merge(float *a, int *b, int low, int high, int mid, float *c, int *d)
{
	int i = low;
	int k = low;
	int j = mid + 1;

	while (i <= mid && j <= high)
	{
		if (a[i] < a[j])
		{
			c[k] = a[i];
			d[k++] = b[i++];
		}
		else
		{
			c[k] = a[j];
			d[k++] = b[j++];
		}
	}

	while (i <= mid)
	{
		c[k] = a[i];
		d[k++] = b[i++];
	}

	while (j <= high)
	{
		c[k] = a[j];
		d[k++] = b[j++];
	}

	for (i = low; i < k; i++)
	{
		a[i] = c[i];
		b[i] = d[i];
	}
}

void mergesort(float *a, int* b, int low, int high, float *tmp_a, int *tmp_b)
{
	int mid;

	if (low < high)
	{
		mid = (low + high) / 2;
		mergesort(a, b, low, mid, tmp_a, tmp_b);
		mergesort(a, b, mid + 1, high, tmp_a, tmp_b);
		merge(a, b, low, high, mid, tmp_a, tmp_b);
	}
	return;
}

float dist(float x1, float y1, float z1, float x2, float y2, float z2) {
	return sqrt(exp2f(x2-x1) + exp2f(y2-y1) + exp2f(z2-z1));
}

PointXYZRGB mediumPoint(PointXYZRGB point_1, PointXYZRGB point_2) {
	PointXYZRGB point;
	point.x = (point_1.x + point_2.x) / 2;
	point.y = (point_1.y + point_2.y) / 2;
	point.z = (point_1.z + point_2.z) / 2;
	point.r = 0;
	point.g = 255;
	point.b = 0;

	return point;
}

void addIntermediatePoint(PointCloud<PointXYZRGB>::Ptr cloud, PointXYZRGB point_1, PointXYZRGB point_2, int step, uint8_t r, uint8_t g, uint8_t b) {
	float delta_x = point_1.x - point_2.x;
	float delta_y = point_1.y - point_2.y;
	float delta_z = point_1.z - point_2.z;

	float step_x = delta_x / step;
	float step_y = delta_y / step;
	float step_z = delta_z / step;

	PointXYZRGB point;
	point.x = point_2.x + step_x;
	point.y = point_2.y + step_y;
	point.z = point_2.z + step_z;
	point.r = r;
	point.g = g;
	point.b = b;

	cloud->push_back(point);

	for (int i = 0; i < step - 2; i++) {
		point.x = point.x + step_x;
		point.y = point.y + step_y;
		point.z = point.z + step_z;
		point.r = r;
		point.g = g;
		point.b = b;
		cloud->push_back(point);
	}

}

void minMaxPoint(PointXYZRGB point) {
	if (point.x < min_x)
		min_x = point.x;
	if (point.x > max_x)
		max_x = point.x;

	if (point.y < min_y)
		min_y = point.y;
	if (point.y > max_y)
		max_y = point.y;

	if (point.z < min_z)
		min_z = point.z;
	if (point.z > max_z)
		max_z = point.z;
}

void updatePoligonPointArray(PointXYZRGB point1, PointXYZRGB point2, PointXYZRGB point3, int poligon_index)
{
	min_poligon_index[poligon_index] = poligon_index;

	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		if (point1.x < point2.x && point1.x < point3.x)
			min_poligon_point[poligon_index] = point1.x;
		else
		{
			if (point2.x < point3.x)
				min_poligon_point[poligon_index] = point2.x;
			else
				min_poligon_point[poligon_index] = point3.x;
		}
	}

	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		if (point1.y < point2.y && point1.y < point3.y)
			min_poligon_point[poligon_index] = point1.y;
		else
		{
			if (point2.y < point3.y)
				min_poligon_point[poligon_index] = point2.y;
			else
				min_poligon_point[poligon_index] = point3.y;
		}
	}
}

void drawLine(PointCloud<PointXYZRGB>::Ptr cloud, PointXYZRGB start_point, Eigen::Vector3f direction, int number_of_point) {
	PointXYZRGB point;
	point.x = start_point.x;
	point.y = start_point.y;
	point.z = start_point.z;
	point.r = 255;
	point.g = 0;
	point.b = 255;

	for (int i = 0; i < number_of_point; i++)
	{
		point.x = point.x + direction[0];
		point.y = point.y + direction[1];
		point.z = point.z + direction[2];
		point.r = 255;
		point.g = 0;
		point.b = 255;
		cloud->push_back(point);
	}

	cloud->width = cloud->points.size();
}

void drawSensor(PointXYZRGB pin_hole, float focal_distance, float sensor_width, float sensor_height, PointCloud<PointXYZRGB>::Ptr cloud) {
	PointXYZRGB sensor_point_00, sensor_point_01, sensor_point_10, sensor_point_11;
	float x_sensor_origin = pin_hole.x - (sensor_height) / 2;
	float y_sensor_origin = pin_hole.y - (sensor_width) / 2;

	sensor_point_00.x = x_sensor_origin;
	sensor_point_00.y = y_sensor_origin;
	sensor_point_00.z = pin_hole.z + focal_distance;

	sensor_point_01.x = x_sensor_origin;
	sensor_point_01.y = y_sensor_origin + sensor_width;
	sensor_point_01.z = pin_hole.z + focal_distance;

	sensor_point_10.x = x_sensor_origin + sensor_height;
	sensor_point_10.y = y_sensor_origin;
	sensor_point_10.z = pin_hole.z + focal_distance;

	sensor_point_11.x = x_sensor_origin + sensor_height;
	sensor_point_11.y = y_sensor_origin + sensor_width;
	sensor_point_11.z = pin_hole.z + focal_distance;

	addIntermediatePoint(cloud, sensor_point_00, sensor_point_01, 100, 0, 255, 0);
	addIntermediatePoint(cloud, sensor_point_00, sensor_point_10, 100, 0, 255, 0);
	addIntermediatePoint(cloud, sensor_point_10, sensor_point_11, 100, 0, 255, 0);
	addIntermediatePoint(cloud, sensor_point_01, sensor_point_11, 100, 0, 255, 0);
}

void initializeMinMaxPoints(PolygonMesh mesh) {
	min_x = min_y = min_z = INT32_MAX;
	max_x = max_y = max_z = INT32_MIN;

	PointCloud<PointXYZ> cloud_mesh;
	PointXYZRGB point_1, point_2, point_3, point, point_m;

	// Metodo veloce per trasformare i vertici della mesh in point cloud
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// ricerca max e min per tutti gli assi
	for (int i = 0; i < mesh.polygons.size(); i++)
	{
		{
			{
				point_1.x = cloud_mesh.points[mesh.polygons[i].vertices[0]].x;
				point_1.y = cloud_mesh.points[mesh.polygons[i].vertices[0]].y;
				point_1.z = cloud_mesh.points[mesh.polygons[i].vertices[0]].z;

				minMaxPoint(point_1);

				point_2.x = cloud_mesh.points[mesh.polygons[i].vertices[1]].x;
				point_2.y = cloud_mesh.points[mesh.polygons[i].vertices[1]].y;
				point_2.z = cloud_mesh.points[mesh.polygons[i].vertices[1]].z;

				minMaxPoint(point_2);

				point_3.x = cloud_mesh.points[mesh.polygons[i].vertices[2]].x;
				point_3.y = cloud_mesh.points[mesh.polygons[i].vertices[2]].y;
				point_3.z = cloud_mesh.points[mesh.polygons[i].vertices[2]].z;

				minMaxPoint(point_3);

				updatePoligonPointArray(point_1, point_2, point_3, i);

			}
		}
	}
}

void initializeLaser(int scanDirection) {
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		laser_point.z = max_z + delta_z;
		laser_point.x = (max_x + min_x) / 2;
		laser_point.y = max_y + (laser_point.z - min_z)*tan(deg2rad(90 - laser_inclination));

		laser_point_2.z = laser_point.z;
		laser_point_2.x = laser_point.x;
		laser_point_2.y = laser_point.y - 2 * distance_laser_sensor;
		
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		laser_point.z = max_z + delta_z;
		laser_point.y = (max_y + min_y) / 2;
		laser_point.x = max_x + (laser_point.z - min_z)*tan(deg2rad(90 - laser_inclination));

		laser_point_2.z = laser_point.z;
		laser_point_2.y = laser_point.y;
		laser_point_2.x = laser_point.x - 2 * distance_laser_sensor;
	}

	laser_point.r = 255;
	laser_point.g = 0;
	laser_point.b = 0;

	laser_point_2.r = 255;
	laser_point_2.g = 0;
	laser_point_2.b = 0;
}

void initializePinHole(int scanDirection, float position) {
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		laser_point.y = position;
		laser_point_2.y = position - 2 * distance_laser_sensor;

	//calcola valori pin hole
		pin_hole.x = laser_point.x;
		pin_hole.y = laser_point.y - distance_laser_sensor;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		laser_point.x = position;
		laser_point_2.x = position - 2 * distance_laser_sensor;

		pin_hole.y = laser_point.y;
		pin_hole.x = laser_point.x - distance_laser_sensor;
		
	}

	pin_hole.z = laser_point.z;
	pin_hole.r = 0;
	pin_hole.g = 0;
	pin_hole.b = 255;
}

void addLaserPoint(PointCloud<PointXYZRGB>::Ptr cloud, int scanDirection) {
	
	initializeLaser(scanDirection);
	cloud->points.push_back(laser_point);

	laser_final_point_left.z = min_z;
	laser_final_point_left.x = laser_point.x - ((max_z - min_z) * tan(deg2rad(laser_aperture / 2)));
	laser_final_point_left.y = laser_point.y - (max_z - min_z) * tan(deg2rad(90 - laser_inclination));
	laser_final_point_left.r = 255;
	laser_final_point_left.g = 0;
	laser_final_point_left.b = 0;
	cloud->points.push_back(laser_final_point_left);

	laser_final_point_right.z = min_z;
	laser_final_point_right.x = laser_point.x + ((max_z - min_z) * tan(deg2rad(laser_aperture / 2)));
	laser_final_point_right.y = laser_point.y - (max_z - min_z) * tan(deg2rad(90 - laser_inclination));
	laser_final_point_right.r = 255;
	laser_final_point_right.g = 0;
	laser_final_point_right.b = 0;
	cloud->points.push_back(laser_final_point_right);

	int num_replicate_point = 50 ;
	addIntermediatePoint(cloud, laser_point, laser_final_point_left, num_replicate_point, 255, 0, 0);
	addIntermediatePoint(cloud, laser_point, laser_final_point_right, num_replicate_point, 255, 0, 0);
	addIntermediatePoint(cloud, laser_final_point_left, laser_final_point_right, num_replicate_point, 255, 0, 0);

	cloud->width = cloud->points.size();
}

Vec3f directionVector(PointXYZRGB source_point, PointXYZRGB destination_point) {
	Vec3f direction;
	direction[0] = (destination_point.x - source_point.x) / (destination_point.z - source_point.z);
	direction[1] = (destination_point.y - source_point.y) / (destination_point.z - source_point.z);
	direction[2] = 1;

	return direction;
}

int triangle_intersection(const Vect3d V1, const Vect3d V2, const Vect3d V3,
						  const Vect3d O, const Vect3d D, float* out, Vect3d &intPoint)
{
	Vect3d e1, e2;  //Edge1, Edge2
	Vect3d P, Q, T;
	float det, inv_det, u, v;
	float t;

	//Find vectors for two edges sharing V1
	//SUB(e1, V2, V1);
	e1 = V2 - V1;
	//SUB(e2, V3, V1);
	e2 = V3 - V1;
	//Begin calculating determinant - also used to calculate u parameter
	//CROSS(P, D, e2);
	P = D.cross(e2);
	//if determinant is near zero, ray lies in plane of triangle
	//det = DOT(e1, P);
	det = e1.dot(P);
	//NOT CULLING
	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	//SUB(T, O, V1);
	T = O - V1;

	//Calculate u parameter and test bound
	//u = DOT(T, P) * inv_det;
	u = T.dot(P) * inv_det;
	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f)
		return 0;

	//Prepare to test v parameter
	//CROSS(Q, T, e1);
	Q = T.cross(e1);

	//Calculate V parameter and test bound
	//v = DOT(D, Q) * inv_det;
	v = D.dot(Q) * inv_det;
	//The intersection lies outside of the triangle
	if (v < 0.f || u + v  > 1.f)
		return 0;

	//t = DOT(e2, Q) * inv_det;
	t = e2.dot(Q) * inv_det;

	if (t > EPSILON) { //ray intersection
		*out = t;

		// get intersection point
		//intersection_point = V1 + u*(V1.cross(V3)) + v*(V1.cross(V2));
		intPoint = O + t*D;

		return 1;
	}

	// No hit, no win
	return 0;
}

// Ritorna la coordinata della direzione di scansione in cui interseca
float rayPlaneLimitIntersection(PointXYZRGB start_point, Eigen::Vector3d direction, float plane_coordinate, int scanDirection) {
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		return direction[1] * (plane_coordinate - start_point.z) / direction[2] + start_point.y;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		return direction[0] * (plane_coordinate - start_point.z) / direction[2] + start_point.x;
	}
	
	return 0;
}

void getPlaneCoefficent(PointXYZRGB laser, Vect3d line_1, Vect3d line_2, Plane* plane) {
	Vect3d plane_normal = line_1.cross(line_2);
	plane->A = plane_normal[0];
	plane->B = plane_normal[1];
	plane->C = plane_normal[2];
	plane->D = -plane_normal[0] * laser.x - plane_normal[1] * laser.y - plane_normal[2] * laser.z;
}

int findStartIndex(float* array_min_points, int array_size, float min_point) {
	int index = 0;

	for (int i = 0; i < array_size; i++)
	{
		if (array_min_points[i] > min_point) {
			index = i - 1;
			break;
		}
	}
	if (index < 0)
		index = 0;

	return index;
}

int findFinalIndex(float* array_min_points, int array_size, float max_point) {
	int index = 0;

	for (int i = array_size-1; i > 0; i--)
	{
		if (array_min_points[i] < max_point) {
			index = i;
			break;
		}
	}

	return index;
}

void findPointsMeshLaserIntersection(const PolygonMesh mesh, const PointXYZRGB laser, 
							   const float density, PointCloud<PointXYZRGB>::Ptr cloudIntersection, int scanDirection, Plane* plane, double laser_number)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	const float MIN_INTERSECTION = VTK_FLOAT_MIN;
	
	int number_of_line = (DIRECTION_TAN_LASER_APERTURE *2) / density;
	//cout << "Numero linee fascio laser: " << number_of_line << endl;

	int d1, d2;
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
		Vect3d line_1(-DIRECTION_TAN_LASER_APERTURE + 0 * density, laser_number * DIRECTION_TAN_LASER_INCLINATION, -1);
		Vect3d line_2(-DIRECTION_TAN_LASER_APERTURE + 10 * density, laser_number * DIRECTION_TAN_LASER_INCLINATION, -1);
		getPlaneCoefficent(laser, line_1, line_2, plane);
		
		//drawLine(cloudIntersection, laser, Eigen::Vector3f(0, -tan(deg2rad(laser_aperture / 2)) + 0 * density, -1), 1000);

	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
		Vect3d line_1(laser_number * DIRECTION_TAN_LASER_INCLINATION, -DIRECTION_TAN_LASER_APERTURE + 0 * density, -1);
		Vect3d line_2(laser_number * DIRECTION_TAN_LASER_INCLINATION, -DIRECTION_TAN_LASER_APERTURE + 1000 * density, -1);

		getPlaneCoefficent(laser, line_1, line_2, plane);
		
		//drawLine(cloudIntersection, laser, Eigen::Vector3f(-tan(deg2rad(laser_aperture / 2)) + 0 * density , 0, -1), 1000);

	}

	Eigen::Vector3d direction_ray_start;
	direction_ray_start[d1] = -DIRECTION_TAN_LASER_APERTURE;
	direction_ray_start[d2] = laser_number * DIRECTION_TAN_LASER_INCLINATION;
	direction_ray_start[2] = -1;
	float min_polygons_coordinate = rayPlaneLimitIntersection(laser, direction_ray_start, min_z, scanDirection);
	float max_polygons_coordinate = rayPlaneLimitIntersection(laser, direction_ray_start, max_z, scanDirection);

	int start_index, final_index;

	if (laser_number == -1)
	{
		start_index = findStartIndex(min_poligon_point, mesh.polygons.size(), min_polygons_coordinate);
		final_index = findFinalIndex(min_poligon_point, mesh.polygons.size(), max_polygons_coordinate);
	}

	else
	{
		start_index = findFinalIndex(min_poligon_point, mesh.polygons.size(), max_polygons_coordinate);
		final_index = findStartIndex(min_poligon_point, mesh.polygons.size(), min_polygons_coordinate);
	}

	//cout << "min_polygons_coordinate: " << min_polygons_coordinate << endl;
	//cout << "max_polygons_coordinate: " << max_polygons_coordinate << endl;
	//cout << "start_index: " << start_index << endl;
	//cout << "final_index: " << final_index << endl;
	cout << "Number of Poligon insercted: " << final_index - start_index << endl;

	#pragma omp parallel for //ordered schedule(dynamic)
	for (int j = 0; j < number_of_line; j++)
	{
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();

		PointXYZ tmp;
		Vertices triangle;
		Vect3d vertex1, vertex2, vertex3;
		Vect3d intersection_point, origin_ray, direction_ray;
		float out;
		PointXYZRGB firstIntersection;

		origin_ray[0] = laser.x;
		origin_ray[1] = laser.y;
		origin_ray[2] = laser.z;

		float i = -DIRECTION_TAN_LASER_APERTURE + j*density;

		firstIntersection.z = MIN_INTERSECTION;

		direction_ray[d1] = i;
		direction_ray[d2] = laser_number * DIRECTION_TAN_LASER_INCLINATION;
		direction_ray[2] = -1;

		

		for (int k = start_index; k < final_index; k++)
		{
			//triangle = mesh.polygons.at(min_poligon_index[k]);
			tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[0]];
			vertex1[0] = tmp.x;
			vertex1[1] = tmp.y;
			vertex1[2] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[1]];;
			vertex2[0] = tmp.x;
			vertex2[1] = tmp.y;
			vertex2[2] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[2]];;
			vertex3[0] = tmp.x;
			vertex3[1] = tmp.y;
			vertex3[2] = tmp.z;

			if (triangle_intersection(vertex1, vertex2, vertex3, origin_ray, direction_ray, &out, intersection_point) != 0)
			{
				if (intersection_point[2] >= firstIntersection.z)
				{

					firstIntersection.x = intersection_point[0];
					firstIntersection.y = intersection_point[1];
					firstIntersection.z = intersection_point[2];
					firstIntersection.r = 255;
					firstIntersection.g = 0;
					firstIntersection.b = 0;

				}
			}
		}

		#pragma omp critical
		{
		//	drawLine(cloudIntersection, laser, Eigen::Vector3f(laser_number*tan(deg2rad(90 - laser_inclination)), i, -1), 1500);


			if (firstIntersection.z > MIN_INTERSECTION)
				cloudIntersection->push_back(firstIntersection);
		}

		//duration<double> timer = high_resolution_clock::now() - start;
		//cout << "Total time cycle ray intersection:" << timer.count() * 1000 << endl;
	}
}

int initializeOpenCL_backup(Triangle* triangle_array, Vec3* output_points, int* output_hits, int array_lenght, Vec3 ray_origin, Vec3 ray_direction) {
	// Length of vectors
	unsigned int n = array_lenght;

	// Host input vectors
	//Triangle* triangle_array;
	// Host output vector
	//Vec3* output_point;
	//int* output_hit;

	// Device input buffers
	//vector<cl::Buffer> mybuffer;
	//mybuffer.push_back(cl::Buffer()); // device_triangle_array
	//mybuffer.push_back(cl::Buffer()); // device_output_point
	//mybuffer.push_back(cl::Buffer()); // device_output_hit

	// Device output buffer
	cl::Buffer device_triangle_array;
	cl::Buffer device_output_points;
	cl::Buffer device_output_hits;

	// Size, in bytes, of each vector
	size_t triangles_size = n*sizeof(Triangle);
	size_t points_size = n*sizeof(Vec3);
	size_t hits_size = n*sizeof(int);

	// Allocate memory for each vector on host
	//triangle_array = new Triangle[n];
	//output_point = new Vec3[n];
	//output_hit = new int[n];

	cl_int err = CL_SUCCESS;
	try {

		// Query platforms
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		// Get list of devices on default platform and create context
		cl_context_properties properties[] =
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);
		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		// Create command queue for first device
		cl::CommandQueue queue(context, devices[0], 0, &err);

		// Create device memory buffers
		device_triangle_array = cl::Buffer(context, CL_MEM_READ_ONLY, triangles_size);
		device_output_points = cl::Buffer(context, CL_MEM_WRITE_ONLY, points_size);
		device_output_hits = cl::Buffer(context, CL_MEM_WRITE_ONLY, hits_size);



		FILE* programHandle;
		size_t kernelSourceSize;
		char *kernelSource;

		// get size of kernel source
		programHandle = fopen("IntersectionTriangle.cl", "r");
		fseek(programHandle, 0, SEEK_END);
		kernelSourceSize = ftell(programHandle);
		rewind(programHandle);

		// read kernel source into buffer
		kernelSource = (char*)malloc(kernelSourceSize + 1);
		kernelSource[kernelSourceSize] = '\0';
		fread(kernelSource, sizeof(char), kernelSourceSize, programHandle);
		fclose(programHandle);

		//Build kernel from source string
		//cl::Program source(kernelSource);
		cl::Program program_ = cl::Program(context, kernelSource);
		program_.build(devices);

		free(kernelSource);

		// Bind memory buffers
		queue.enqueueWriteBuffer(device_triangle_array, CL_TRUE, 0, triangles_size, triangle_array);

		// Create kernel object
		cl::Kernel kernel(program_, "RayTriangleIntersection", &err);

		// Bind kernel arguments to kernel
		kernel.setArg(0, device_triangle_array);
		kernel.setArg(1, device_output_points);
		kernel.setArg(2, device_output_hits);
		kernel.setArg(3, n);
		kernel.setArg(4, ray_origin);
		kernel.setArg(5, ray_direction);

		// Number of work items in each local work group
		cl::NDRange localSize(64);
		// Number of total work items - localSize must be devisor
		cl::NDRange globalSize((int)(ceil(n / (float)64) * 64));

		/*high_resolution_clock::time_point start;
		start = high_resolution_clock::now();*/

		// Enqueue kernel
		cl::Event event;
		queue.enqueueNDRangeKernel(
			kernel,
			cl::NullRange,
			globalSize,
			localSize,
			NULL,
			&event);

		// Block until kernel completion
		event.wait();

		//duration<double> timer = high_resolution_clock::now() - start;
		//cout << "Total time Intersection OpenCL:" << timer.count() * 1000 << endl;

		// Read back device_output_point, device_output_hit
		queue.enqueueReadBuffer(device_output_points, CL_TRUE, 0, points_size, output_points);
		queue.enqueueReadBuffer(device_output_hits, CL_TRUE, 0, hits_size, output_hits);
	}
	catch (...) {
		cout << "Errore OpenCL " << endl;

		return -1;

	}


	// Release host memory
	//delete(triangle_array);
	//delete(output_point);
	//delete(output_hit);

	//mybuffer.clear();

	return 0;
}

void prepareDataForOpenCL(const PolygonMesh mesh, Triangle* triangles) {
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	PointXYZ tmp;
	

	for (int k = 0; k < mesh.polygons.size(); k++)
	{

		tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[0]];
		triangles[k].vertex1.points[X] = tmp.x;
		triangles[k].vertex1.points[Y] = tmp.y;
		triangles[k].vertex1.points[Z] = tmp.z;

		tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[1]];
		triangles[k].vertex2.points[X] = tmp.x;
		triangles[k].vertex2.points[Y] = tmp.y;
		triangles[k].vertex2.points[Z] = tmp.z;

		tmp = meshVertices.points[mesh.polygons[min_poligon_index[k]].vertices[2]];
		triangles[k].vertex3.points[X] = tmp.x;
		triangles[k].vertex3.points[Y] = tmp.y;
		triangles[k].vertex3.points[Z] = tmp.z;
	}
}

int initializeOpenCL(OpenCLDATA* openCLData, Triangle* triangle_array, int array_lenght) {

	cl_int err = CL_SUCCESS;
	try {

		// Query platforms
		cl::Platform::get(&openCLData->platforms);
		if (openCLData->platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;
		}

		// Get list of devices on default platform and create context
		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(openCLData->platforms[0])(), 0 };
		openCLData->context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
		//openCLData->context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
		openCLData->devices = openCLData->context.getInfo<CL_CONTEXT_DEVICES>();

		// Create command queue for first device
		openCLData->queue = cl::CommandQueue(openCLData->context, openCLData->devices[0], 0, &err);

		FILE* programHandle;
		size_t kernelSourceSize;
		char *kernelSource;
		
		// get size of kernel source
		programHandle = fopen("IntersectionTriangle.cl", "r");
		fseek(programHandle, 0, SEEK_END);
		kernelSourceSize = ftell(programHandle);
		rewind(programHandle);

		// read kernel source into buffer
		kernelSource = (char*)malloc(kernelSourceSize + 1);
		kernelSource[kernelSourceSize] = '\0';
		fread(kernelSource, sizeof(char), kernelSourceSize, programHandle);
		fclose(programHandle);

		//Build kernel from source string
		//cl::Program source(kernelSource);
		openCLData->program_ = cl::Program(openCLData->context, kernelSource);
		openCLData->program_.build(openCLData->devices);

		free(kernelSource);

		// Size, in bytes, of each vector
		openCLData->triangles_size = array_lenght*sizeof(Triangle);
		openCLData->points_size = array_lenght*sizeof(Vec3);
		openCLData->hits_size = array_lenght*sizeof(uchar);

		// Create device memory buffers
		openCLData->device_triangle_array = cl::Buffer(openCLData->context, CL_MEM_READ_ONLY, openCLData->triangles_size);
		openCLData->device_output_points = cl::Buffer(openCLData->context, CL_MEM_WRITE_ONLY, openCLData->points_size);
		openCLData->device_output_hits = cl::Buffer(openCLData->context, CL_MEM_WRITE_ONLY, openCLData->hits_size);

		// Bind memory buffers
		openCLData->queue.enqueueWriteBuffer(openCLData->device_triangle_array, CL_TRUE, 0, openCLData->triangles_size, triangle_array);

		// Create kernel object
		openCLData->kernel = cl::Kernel(openCLData->program_, "RayTriangleIntersection", &err);

		// Bind kernel arguments to kernel
		openCLData->kernel.setArg(0, openCLData->device_triangle_array);
		openCLData->kernel.setArg(1, openCLData->device_output_points);
		openCLData->kernel.setArg(2, openCLData->device_output_hits);

	}
	catch (...) {
		cout << "Errore OpenCL " << endl;

		return -1;

	}

	return 0;
}

inline int computeOpenCL(OpenCLDATA* openCLData, Vec3* output_points, uchar* output_hits, int start_index, int array_lenght, Vec3 ray_origin, Vec3 ray_direction) {

	//high_resolution_clock::time_point start;
	//start = high_resolution_clock::now();

	cl_int err = CL_SUCCESS;

	// Device output buffer
	//cl::Buffer device_triangle_array;
	//cl::Buffer device_output_points;
	//cl::Buffer device_output_hits;

	// Size, in bytes, of each vector
	//size_t triangles_size = array_lenght*sizeof(Triangle);
	//size_t points_size = array_lenght*sizeof(Vec3);
	//size_t hits_size = array_lenght*sizeof(int);

	// Create device memory buffers
	//device_triangle_array = cl::Buffer(openCLData->context, CL_MEM_READ_ONLY, triangles_size);
	//device_output_points = cl::Buffer(openCLData->context, CL_MEM_WRITE_ONLY, points_size);
	//device_output_hits = cl::Buffer(openCLData->context, CL_MEM_WRITE_ONLY, hits_size);

	// Bind memory buffers
	//openCLData->queue.enqueueWriteBuffer(device_triangle_array, CL_TRUE, 0, triangles_size, triangle_array);

	
	openCLData->kernel.setArg(3, start_index);
	openCLData->kernel.setArg(4, array_lenght);
	openCLData->kernel.setArg(5, ray_origin);
	openCLData->kernel.setArg(6, ray_direction);

	// Number of work items in each local work group
#define LOCAL_SIZE 128
	cl::NDRange localSize(LOCAL_SIZE, 1, 1);
	float m64 = 64;
	// Number of total work items - localSize must be devisor
	int global_size = (int)(ceilf((array_lenght/256) / LOCAL_SIZE) * LOCAL_SIZE);
	//cout << "global_size " << global_size << endl;
	cl::NDRange globalSize(global_size, 1, 1);

	// Enqueue kernel
	cl::Event event;
	openCLData->queue.enqueueNDRangeKernel(
		openCLData->kernel,
		cl::NullRange,
		globalSize,
		localSize,
		NULL,
		&event);

	// Block until kernel completion
	event.wait();

	// Read back device_output_point, device_output_hit
	openCLData->queue.enqueueReadBuffer(openCLData->device_output_points, CL_TRUE, 0, openCLData->points_size, output_points);
	openCLData->queue.enqueueReadBuffer(openCLData->device_output_hits, CL_TRUE, 0, openCLData->hits_size, output_hits);

	//duration<double> timer = high_resolution_clock::now() - start;
	//cout << "Buffer output copied OpenCL:" << timer.count() * 1000 << endl;

	return 0;
}

inline void findPointsMeshLaserIntersectionOpenCL(OpenCLDATA* openCLData, Triangle* all_triangles, Vec3* output_points, uchar* output_hits,
	const PolygonMesh mesh, const PointXYZRGB laser,
	const float density, PointCloud<PointXYZRGB>::Ptr cloudIntersection, int scanDirection, Plane* plane, double laser_number)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	const float MIN_INTERSECTION = VTK_FLOAT_MIN;

	int number_of_line = (DIRECTION_TAN_LASER_APERTURE * 2) / density;
	//cout << "Numero linee fascio  laser: " << number_of_line << endl;

	int d1, d2;
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
		Vect3d line_1(-DIRECTION_TAN_LASER_APERTURE + 0 * density, laser_number * DIRECTION_TAN_LASER_INCLINATION, -1);
		Vect3d line_2(-DIRECTION_TAN_LASER_APERTURE + 10 * density, laser_number * DIRECTION_TAN_LASER_INCLINATION, -1);
		getPlaneCoefficent(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(0, -tan(deg2rad(laser_aperture / 2)) + 0 * density, -1), 1000);

	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
		Vect3d line_1(laser_number * DIRECTION_TAN_LASER_INCLINATION, -DIRECTION_TAN_LASER_APERTURE + 0 * density, -1);
		Vect3d line_2(laser_number * DIRECTION_TAN_LASER_INCLINATION, -DIRECTION_TAN_LASER_APERTURE + 1000 * density, -1);

		getPlaneCoefficent(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(-tan(deg2rad(laser_aperture / 2)) + 0 * density , 0, -1), 1000);

	}

	Eigen::Vector3d direction_ray_start;
	direction_ray_start[d1] = -DIRECTION_TAN_LASER_APERTURE; 
	direction_ray_start[d2] = laser_number * DIRECTION_TAN_LASER_INCLINATION;
	direction_ray_start[2] = -1;
	float min_polygons_coordinate = rayPlaneLimitIntersection(laser, direction_ray_start, min_z, scanDirection);
	float max_polygons_coordinate = rayPlaneLimitIntersection(laser, direction_ray_start, max_z, scanDirection);

	int start_index, final_index;

	if (laser_number == -1)
	{
		start_index = findStartIndex(min_poligon_point, mesh.polygons.size(), min_polygons_coordinate);
		final_index = findFinalIndex(min_poligon_point, mesh.polygons.size(), max_polygons_coordinate);
	}
	else
	{
		start_index = findFinalIndex(min_poligon_point, mesh.polygons.size(), max_polygons_coordinate);
		final_index = findStartIndex(min_poligon_point, mesh.polygons.size(), min_polygons_coordinate);
	}

	//cout << "min_polygons_coordinate: " << min_polygons_coordinate << endl;
	//cout << "max_polygons_coordinate: " << max_polygons_coordinate << endl;
	//cout << "start_index: " << start_index << endl;
	//cout << "final_index: " << final_index << endl;
	cout << "Number of Poligon insercted: " << final_index - start_index << endl;

	for (int j = 0; j < number_of_line; j++)
	{
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();

		PointXYZ tmp;
		Vertices triangle;
		Vect3d vertex1, vertex2, vertex3;
		Vect3d intersection_point;
		Vec3 ray_origin, ray_direction;
		float out;
		PointXYZRGB firstIntersection;

		ray_origin.points[X] = laser.x;
		ray_origin.points[Y] = laser.y;
		ray_origin.points[Z] = laser.z;

		float i = -DIRECTION_TAN_LASER_APERTURE + j*density;

		firstIntersection.z = MIN_INTERSECTION;

		if (scanDirection == DIRECTION_SCAN_AXIS_Y)
			ray_direction.points[X] = i;
		else
			ray_direction.points[Y] = i;
		if (scanDirection == DIRECTION_SCAN_AXIS_X)
			ray_direction.points[X] = laser_number * DIRECTION_TAN_LASER_INCLINATION;
		else
			ray_direction.points[Y] = laser_number * DIRECTION_TAN_LASER_INCLINATION;

		ray_direction.points[Z] = -1;

		int diff = final_index - start_index;
		//cout << "final_index - start_index: " << diff << endl;

		//Triangle* triangles = all_triangles + start_index;
		//Vec3* output_points = new Vec3[diff];
		//int* output_hits = new int[diff];

		computeOpenCL(openCLData, output_points, output_hits, start_index, diff, ray_origin, ray_direction);
		//initializeOpenCL_backup(triangles, output_points, output_hits, diff, ray_origin, ray_direction);
		
		/*for (int k = 0; k < diff; k++)
		{
			Triangle triangle = triangles[k];

			output_hits[k] = triangleIntersection_C(triangle.vertex1, triangle.vertex2, triangle.vertex3, ray_origin, ray_direction, &output_points[k]);

		}*/

		//int hit_number = 0;
		//cout << endl;
		for (int h = start_index; h < final_index; h++)
		{
			if (output_hits[h] == 1)
			{
				//++hit_number;

				if (output_points[h].points[Z] >= firstIntersection.z)
				{
					//cout << "hit point:" << output_points[h].x << "," << output_points[h].y << "," << output_points[h].z << endl;

					firstIntersection.x = output_points[h].points[X];
					firstIntersection.y = output_points[h].points[Y];
					firstIntersection.z = output_points[h].points[Z];
					firstIntersection.r = 255;
					firstIntersection.g = 0;
					firstIntersection.b = 0;

				}

			}
		}

		if (firstIntersection.z > MIN_INTERSECTION)
			cloudIntersection->push_back(firstIntersection);

		//cout << "hit_number: " << hit_number << endl;

		//delete(triangles);
		//delete(output_points);
		//delete(output_hits);

		//duration<double> timer3 = high_resolution_clock::now() - start;
		//cout << "Total time cycle ray intersection OpenCL:" << timer3.count() * 1000 << endl;
	}
}

void sensorPointProjection(float focal_distance, float sensor_height, float sensor_width, PointCloud<PointXYZRGB>::Ptr cloud_intersection, PointCloud<PointXYZRGB>::Ptr cloud_projection)
{
	PointXYZRGB tmp, p, center;
	center.x = pin_hole.x;
	center.y = pin_hole.y;
	center.z = pin_hole.z + focal_distance;

	for (int i = 0; i < cloud_intersection->points.size(); i++)
	{
		tmp.x = cloud_intersection->points[i].x;
		tmp.y = cloud_intersection->points[i].y;
		tmp.z = cloud_intersection->points[i].z;

		// calcolo le coordinate della proiezione sul piano z = pin_hole.z + distanza focale
		p.x = (pin_hole.x - tmp.x) * (center.z - tmp.z) / (pin_hole.z - tmp.z) + tmp.x;
		p.y = (pin_hole.y - tmp.y) * (center.z - tmp.z) / (pin_hole.z - tmp.z) + tmp.y;
		p.z = center.z;
		p.r = 255;
		p.g = 0;
		p.b = 0;

		// controllo se il punto è dentro al rettangolo del sensore e se lo è lo aggiungo
		if (scanDirection == DIRECTION_SCAN_AXIS_X) {
			if (p.x < center.x + sensor_height / 2 && p.x > center.x - sensor_height / 2 &&
				p.y < center.y + sensor_width / 2 && p.y > center.y - sensor_width / 2)
			{
				cloud_projection->push_back(p);
			}
		}
		if (scanDirection == DIRECTION_SCAN_AXIS_Y) {
			if (p.x < center.x + sensor_width / 2 && p.x > center.x - sensor_width / 2 &&
				p.y < center.y + sensor_height / 2 && p.y > center.y - sensor_height / 2)
			{
				cloud_projection->push_back(p);
			}
		}
	}
}

int drawLaserImage(PointXYZRGB pin_hole, Mat* image_out, int sensor_pixel_height, int sensor_pixel_width, PointCloud<PointXYZRGB>::Ptr cloud_projection) {
	int image_point_added = 0;
	Mat image(sensor_pixel_height, sensor_pixel_width, CV_8UC3);
	float sensor_width = sensor_pixel_width * PIXEL_DIMENSION;
	float sensor_height = sensor_pixel_height * PIXEL_DIMENSION;
	float x_sensor_origin, y_sensor_origin;

	// inizializza l'immagine bianca
	for (int i = 0; i < sensor_pixel_height; i++)
		for (int j = 0; j < sensor_pixel_width; j++) {
			image.at<Vec3b>(i, j)[0] = 255;
			image.at<Vec3b>(i, j)[1] = 255;
			image.at<Vec3b>(i, j)[2] = 255;
		}

	if (scanDirection == DIRECTION_SCAN_AXIS_X) {
		x_sensor_origin = pin_hole.x - (sensor_height) / 2;
		y_sensor_origin = pin_hole.y - (sensor_width) / 2;

		for (int i = 0; i < cloud_projection->points.size(); i++)
		{
			float x, y;
			x = cloud_projection->points[i].x;
			y = cloud_projection->points[i].y;

			int x_pos = ((x - x_sensor_origin) / PIXEL_DIMENSION);
			int y_pos = ((y - y_sensor_origin) / PIXEL_DIMENSION);

			if (x_pos >= 0 && x_pos < sensor_pixel_height && y_pos >= 0 && y_pos < sensor_pixel_width) {

				Vec3b & color = image.at<Vec3b>(x_pos, y_pos);
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
				image_point_added++;
			}
		}
	}

	if (scanDirection == DIRECTION_SCAN_AXIS_Y) {
		x_sensor_origin = pin_hole.x + (sensor_width) / 2;
		y_sensor_origin = pin_hole.y - (sensor_height) / 2;

		for (int i = 0; i < cloud_projection->points.size(); i++)
		{
			float x, y;
			x = cloud_projection->points[i].x;
			y = cloud_projection->points[i].y;

			int y_pos = ((x_sensor_origin - x) / PIXEL_DIMENSION);
			int x_pos = ((y - y_sensor_origin) / PIXEL_DIMENSION);

			if (x_pos >= 0 && x_pos < sensor_pixel_height && y_pos >= 0 && y_pos < sensor_pixel_width) {

				Vec3b & color = image.at<Vec3b>(x_pos, y_pos);
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
				image_point_added++;
			}
		}
	}
	*image_out = image;

	return image_point_added;
}

void getCameraFrameMauro(PointXYZRGB pin_hole, Mat* image_out, int sensor_pixel_height, int sensor_pixel_width, PointCloud<PointXYZRGB>::Ptr cloud_intersection) {
	PointCloud<PointXYZRGB>::Ptr cloud_projection(new PointCloud<PointXYZRGB>);
	sensorPointProjection(focal_distance, sensor_height, sensor_width, cloud_intersection, cloud_projection);
	drawLaserImage(pin_hole, image_out, sensor_pixel_height, sensor_pixel_width, cloud_projection);
	cloud_projection->~PointCloud();
}

void generatePointCloudFromImageMauro(Plane* plane1, Plane* plane2, Mat* image, PointCloud<PointXYZ>::Ptr cloud_out) {
	PointXYZ point;
	float dx, dy, dz;  // vettore direzionale retta punto-pin_hole
	float x_sensor_origin, y_sensor_origin;

	if (scanDirection == DIRECTION_SCAN_AXIS_X) {
		x_sensor_origin = pin_hole.x - (sensor_height) / 2;
		y_sensor_origin = pin_hole.y - (sensor_width) / 2;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_Y) {
		x_sensor_origin = pin_hole.x + (sensor_width) / 2;
		y_sensor_origin = pin_hole.y - (sensor_height) / 2;
	}

	Mat cameraMatrix;

	// parametri intrinseci della telecamera
	cameraMatrix = Mat::zeros(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = 4615.04; // Fx
	cameraMatrix.at<double>(1, 1) = 4615.51; // Fy
	cameraMatrix.at<double>(0, 2) = 1113.41; // Cx
	cameraMatrix.at<double>(1, 2) = 480.016; // Cy
	cameraMatrix.at<double>(2, 2) = 1;

	Mat image_undistort;
	undistort(*image, image_undistort, cameraMatrix, Mat::zeros(8, 1, CV_64F));
	flip(image_undistort, *image, 0); // altrimenti la cloud viene rovescia

									  // Creo la point cloud del sensore a partire dall'immagine
	for (int i = 0; i < image->rows / 2; i++)
	{
		for (int j = 0; j < image->cols; j++)
		{
			Vec3b & color = image->at<Vec3b>(i, j);
			// controlla che sia colorato il pixel dell'immagine
			if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
				// Posiziono i punti dell'immagine nel sensore virtuale
				if (scanDirection == DIRECTION_SCAN_AXIS_X) {
					point.x = i * PIXEL_DIMENSION + x_sensor_origin;
					point.y = j * PIXEL_DIMENSION + y_sensor_origin;
				}
				if (scanDirection == DIRECTION_SCAN_AXIS_Y) {
					point.x = x_sensor_origin - j * PIXEL_DIMENSION;
					point.y = i * PIXEL_DIMENSION + y_sensor_origin;
				}
				point.z = pin_hole.z + focal_distance;

				dx = pin_hole.x - point.x;
				dy = pin_hole.y - point.y;
				dz = pin_hole.z - point.z;

				// Proietto il punto del sensore sul piano laser passando dal pin hole
				float t = -(plane1->A * point.x + plane1->B * point.y + plane1->C * point.z + plane1->D) / (plane1->A * dx + plane1->B * dy + plane1->C * dz);
				point.x = dx * t + point.x;
				point.y = dy * t + point.y;
				point.z = dz * t + point.z;
				cloud_out->push_back(point);
			}
		}
	}

	for (int i = image->rows / 2; i < image->rows; i++)
	{
		for (int j = 0; j < image->cols; j++)
		{
			Vec3b & color = image->at<Vec3b>(i, j);
			// controlla che sia colorato il pixel dell'immagine
			if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
				// Posiziono i punti dell'immagine nel sensore virtuale
				if (scanDirection == DIRECTION_SCAN_AXIS_X) {
					point.x = i * PIXEL_DIMENSION + x_sensor_origin;
					point.y = j * PIXEL_DIMENSION + y_sensor_origin;
				}
				if (scanDirection == DIRECTION_SCAN_AXIS_Y) {
					point.x = x_sensor_origin - j * PIXEL_DIMENSION;
					point.y = i * PIXEL_DIMENSION + y_sensor_origin;
				}
				point.z = pin_hole.z + focal_distance;

				dx = pin_hole.x - point.x;
				dy = pin_hole.y - point.y;
				dz = pin_hole.z - point.z;

				// Proietto il punto del sensore sul piano laser passando dal pin hole
				float t = -(plane2->A * point.x + plane2->B * point.y + plane2->C * point.z + plane2->D) / (plane2->A * dx + plane2->B * dy + plane2->C * dz);
				point.x = dx * t + point.x;
				point.y = dy * t + point.y;
				point.z = dz * t + point.z;
				cloud_out->push_back(point);
			}
		}
	}
}

int main(int argc, char** argv)
{
	// Load STL file as a PolygonMesh
	PolygonMesh mesh;
	PointCloud<PointXYZRGB>::Ptr cloud_projection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	Mat image, image2;
	OpenCLDATA openCLData;
	Triangle* all_triangles;

	if (io::loadPolygonFileSTL("../dataset/bin1.stl", mesh) == 0)
	{
		PCL_ERROR("Failed to load STL file\n");
		return -1;
	}
	cout << mesh.polygons.size() << " Processing point cloud... " << endl;

	// Trova i punti di min e max per tutti gli assi della mesh
	min_poligon_point = new float[mesh.polygons.size()]; // array per salvare i punti più a sx dei poligoni
	min_poligon_index = new int[mesh.polygons.size()];   // array per salvare l'indice di tali punti
	initializeMinMaxPoints(mesh);

	cout << "min_x:" << min_x << " max_x:" << max_x << endl;
	cout << "min_y:" << min_y << " max_y:" << max_y << endl;
	cout << "min_z:" << min_z << " max_z:" << max_z << endl;

	// Ordino gli array per una ricerca più efficiente dei poligoni
	float *tmp_a = new float[mesh.polygons.size()]; // li creo qui fuori perché creandoli ogni volta nella ricorsione
	int *tmp_b = new int[mesh.polygons.size()];     // c'è un crash dovuto alla ricorsione dell'operatore new
	mergesort(min_poligon_point, min_poligon_index, 0, mesh.polygons.size() - 1, tmp_a, tmp_b);
	delete[] tmp_a, tmp_b; // elimino gli array temporanei

	// Inizializza il laser
	initializeLaser(scanDirection);

	cout << "Laser_point x:" << laser_point.x << " y:" << laser_point.y << " z:" << laser_point.z << endl;

	// lo sposto già un po' più avanti per un bug che interseca tutti i triangoli

	// ATTENZIONE: al verso di scansione
	float position_step;
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		position_step = laser_point.x - 1180; //- 1;
	}
	else if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		position_step = laser_point.y - 550;
	}

	cout << "position_step:" << position_step << endl;

	// Questo valore varia da 0,2 a 10 frame per mm
	float increment_value = scan_speed / camera_fps;

	PointCloud<PointXYZ>::Ptr cloudGenerate(new PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr cloudOut(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_test(new PointCloud<PointXYZRGB>);

	Mat image3(sensor_pixel_width, sensor_pixel_height, CV_8UC3);
	// inizializza l'immagine bianca
	for (int i = 0; i < image3.rows; i++)
		for (int j = 0; j < image3.cols; j++) {
			image3.at<Vec3b>(i, j)[0] = 255;
			image3.at<Vec3b>(i, j)[1] = 255;
			image3.at<Vec3b>(i, j)[2] = 255;
		}

	// OpenCL Loading
	int array_size_64 = (int)(ceilf(mesh.polygons.size() / 64) * 64);
	int size_array = mesh.polygons.size();
	all_triangles = new Triangle[size_array];
	Vec3* output_points = new Vec3[size_array];
	uchar* output_hits = new uchar[size_array];
	prepareDataForOpenCL(mesh, all_triangles);
	initializeOpenCL(&openCLData, all_triangles, size_array);

	for (int z = 0; z < 1; z++)
	{

		cout << "Z->" << z << " ";
		cout << "position_step: " << position_step << endl;

		// Inizializza il Pin Hole e sposta anche la posizione del laser
		initializePinHole(scanDirection, position_step);
		position_step -= increment_value;

		cout << "Laser_point 1 x:" << laser_point.x << " y:" << laser_point.y << " z:" << laser_point.z << endl;
		cout << "Laser_point 2 x:" << laser_point.x << " y:" << laser_point.y << " z:" << laser_point.z << endl;

		Plane plane1, plane2;

		high_resolution_clock::time_point start;
		start = high_resolution_clock::now();

		// cerca i punti di insersezione del raggio laser
		//findPointsMeshLaserIntersection(mesh, laser_point, RAY_DENSITY, cloud_intersection, scanDirection, &plane1, LASER_1);

		//findPointsMeshLaserIntersection(mesh, laser_point, RAY_DENSITY, cloud_intersection, scanDirection, &plane1, LASER_1);
		//findPointsMeshLaserIntersection(mesh, laser_point_2, RAY_DENSITY, cloud_intersection, scanDirection, &plane2, LASER_2);

		findPointsMeshLaserIntersectionOpenCL(&openCLData, all_triangles, output_points, output_hits, mesh, laser_point, RAY_DENSITY, cloud_intersection, scanDirection, &plane1, LASER_1);
		//findPointsMeshLaserIntersectionOpenCL(&openCLData, all_triangles, output_points, output_hits, mesh, laser_point_2, RAY_DENSITY, cloud_intersection, scanDirection, &plane2, LASER_2);

		duration<double> timer2 = high_resolution_clock::now() - start;
		cout << "Total time Intersection:" << timer2.count() * 1000 << endl;

		//****************** Converto la point cloud in un immagine **********************

		getCameraFrameMauro(pin_hole, &image2, sensor_pixel_height, sensor_pixel_width, cloud_intersection);

		flip(image2, image2, 0);
		generatePointCloudFromImageMauro(&plane2, &plane1, &image2, cloudOut);


		for (int i = 0; i < cloud_intersection->size(); i++)
		{
			cloud_test->push_back(cloud_intersection->at(i));
		}

		cloud_intersection->~PointCloud();
		cloudGenerate->~PointCloud();  
		cloud_projection->~PointCloud();
	}
	cout << "Punti cloud_test " << cloud_test->points.size();


	// Create a PCLVisualizer
	/*visualization::PCLVisualizer viewer("Mesh");
	viewer.addCoordinateSystem(100, "mesh");
	viewer.addPolygonMesh(mesh, "mesh");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb3(cloud_intersection);
	viewer.addPointCloud<PointXYZRGB>(cloud_intersection, rgb3, "cloudScan");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb4(cloud_projection);
	viewer.addPointCloud<PointXYZRGB>(cloud_projection, rgb4, "cloudProj");
	viewer.spin();*/

	visualization::PCLVisualizer viewer2("Viewer2");
	viewer2.addCoordinateSystem(0.1, "viewer2");
	viewer2.addPointCloud<PointXYZ>(cloudOut, "cloudGen");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb4(cloud_test);
	viewer2.addPointCloud<PointXYZRGB>(cloud_test, rgb4, "cloudTest");
	viewer2.spin();


	return 0;
}


