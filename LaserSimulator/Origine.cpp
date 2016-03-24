/*
* computePointCloud.cpp
* Created on: 10/12/2015
* Last Update: 21/12/2015
* Author: Nicola Rigato 1110346
*
*/

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

using namespace cv;
using namespace std;
using namespace pcl;

#define EPSILON 0.000001
#define PIXEL_DIMENSION 0.0055 // mm

#define DIRECTION_SCAN_AXIS_X 0
#define DIRECTION_SCAN_AXIS_Y 1

// MIN MAX POINT
float min_x, max_x;
float min_y, max_y;
float min_z, max_z;

PointXYZRGB laser_point, laser_final_point_left, laser_final_point_right, pin_hole;

int scanDirection = DIRECTION_SCAN_AXIS_X;
float distance_laser_sensor = 600 ; //600
float laser_aperture = 45.0;
float laser_inclination = 60.0;
float delta_z = 600;
int default_number_samples = 10000000;

int sensor_pixel_width = 2024;
int sensor_pixel_height = 1088;
float focal_distance = 25;

float sensor_width = sensor_pixel_width * PIXEL_DIMENSION;
float sensor_height = sensor_pixel_height * PIXEL_DIMENSION;

float *min_poligon_point;
int   *min_poligon_index;

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
		laser_point.r = 255;
		laser_point.g = 0;
		laser_point.b = 0;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		laser_point.z = max_z + delta_z;
		laser_point.y = (max_y + min_y) / 2;
		laser_point.x = max_x + (laser_point.z - min_z)*tan(deg2rad(90 - laser_inclination));
		laser_point.r = 255;
		laser_point.g = 0;
		laser_point.b = 0;
	}
}

void initializePinHole(int scanDirection, int position) {
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		laser_point.y -= position; // solo per test... da rimuovere

	//calcola valori pin hole
		pin_hole.x = laser_point.x;
		pin_hole.y = laser_point.y - distance_laser_sensor;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		laser_point.x -= position; // solo per test... da rimuovere

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

int triangle_intersection(const Eigen::Vector3d V1, const Eigen::Vector3d V2, const Eigen::Vector3d V3, 
						  const Eigen::Vector3d O, const Eigen::Vector3d D, float* out, Eigen::Vector3d &intPoint)
{
	Eigen::Vector3d e1, e2;  //Edge1, Edge2
	Eigen::Vector3d P, Q, T;
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

float rayPlaneIntersection(PointXYZRGB start_point, Eigen::Vector3d direction, float plane_coordinate, int scanDirection) {
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		return direction[1] * (plane_coordinate - start_point.z) / direction[2] + start_point.y;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		return direction[0] * (plane_coordinate - start_point.z) / direction[2] + start_point.x;
	}

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
							   const float density, PointCloud<PointXYZRGB>::Ptr cloudIntersection, int scanDirection)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	const float MIN_INTERSECTION = VTK_FLOAT_MIN;
	
	int number_of_line = (tan(deg2rad(laser_aperture / 2))*2) / density;
	cout << "Numero linee fascio laser: " << number_of_line << endl;

	int d1, d2;
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
	}

	Eigen::Vector3d direction_ray_start;
	direction_ray_start[d1] = -tan(deg2rad(laser_aperture / 2));
	direction_ray_start[d2] = -tan(deg2rad(90 - laser_inclination));
	direction_ray_start[2] = -1;
	float min_polygons_coordinate = rayPlaneIntersection(laser_point, direction_ray_start, min_z, scanDirection);
	float max_polygons_coordinate = rayPlaneIntersection(laser_point, direction_ray_start, max_z, scanDirection);
	int start_index = findStartIndex(min_poligon_point, mesh.polygons.size(), min_polygons_coordinate);
	int final_index = findFinalIndex(min_poligon_point, mesh.polygons.size(), max_polygons_coordinate);

	cout << "min_polygons_coordinate: " << min_polygons_coordinate << endl;
	cout << "max_polygons_coordinate: " << max_polygons_coordinate << endl;
	cout << "start_index: " << start_index << endl;
	cout << "final_index: " << final_index << endl;
	cout << "Number of Poligon insercted: " << final_index - start_index << endl;

	#pragma omp parallel for //ordered schedule(dynamic)
	for (int j = 0; j < number_of_line; j++)
	{
		PointXYZ tmp;
		Vertices triangle;
		Eigen::Vector3d vertex1, vertex2, vertex3;
		Eigen::Vector3d intersection_point, origin_ray, direction_ray;
		float out;
		PointXYZRGB firstIntersection;

		origin_ray[0] = laser.x;
		origin_ray[1] = laser.y;
		origin_ray[2] = laser.z;

		float i = -tan(deg2rad(laser_aperture / 2)) + j*density;

		firstIntersection.z = MIN_INTERSECTION;

		direction_ray[d1] = i;
		direction_ray[d2] = -tan(deg2rad(90 - laser_inclination));
		direction_ray[2] = -1;

		for (int k = start_index; k < final_index; k++)
		{
			triangle = mesh.polygons.at(min_poligon_index[k]);
			tmp = meshVertices.points[triangle.vertices[0]];
			vertex1[0] = tmp.x;
			vertex1[1] = tmp.y;
			vertex1[2] = tmp.z;

			tmp = meshVertices.points[triangle.vertices[1]];
			vertex2[0] = tmp.x;
			vertex2[1] = tmp.y;
			vertex2[2] = tmp.z;

			tmp = meshVertices.points[triangle.vertices[2]];
			vertex3[0] = tmp.x;
			vertex3[1] = tmp.y;
			vertex3[2] = tmp.z;

			if (triangle_intersection(vertex1, vertex2, vertex3, origin_ray, direction_ray, &out, intersection_point) != 0) {
				if (intersection_point[2] >= firstIntersection.z) {

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
			//drawLine(cloudIntersection, laser_point, Eigen::Vector3f(-tan(deg2rad(90 - laser_inclination)), i, -1), 1500);

			if (firstIntersection.z > MIN_INTERSECTION)
				cloudIntersection->push_back(firstIntersection);
		}
	}
}

void findPointsOctreeLaserIntersection(octree::OctreePointCloudSearch<PointXYZRGB> tree, PointCloud<PointXYZRGB>::Ptr cloud_scan, 
						PointCloud<PointXYZRGB>::Ptr cloud_out, int intersect_points) {
	vector<int> indices;
	intersect_points = 0;

	for (float i = -tan(deg2rad(laser_aperture / 2)); i < tan(deg2rad(laser_aperture / 2)); i += 0.0001)
	{
		if (tree.getIntersectedVoxelIndices(Eigen::Vector3f(laser_point.x, laser_point.y, laser_point.z), Eigen::Vector3f(i, -tan(deg2rad(90 - laser_inclination)), -1), indices) > 0) 
		{

			// preleva solo il primo punto intersecato dalla retta, quello più superficiale
			cloud_out->points[indices[0]].r = 0;
			cloud_out->points[indices[0]].g = 255;
			cloud_out->points[indices[0]].b = 0;
			cloud_scan->points.push_back(cloud_out->points[indices[0]]);
			intersect_points++;
			//}
		}
		drawLine(cloud_scan, laser_point, Eigen::Vector3f(i, -tan(deg2rad(90 - laser_inclination)), -1), 500);

	}

	cloud_scan->height = cloud_scan->points.size();
	cloud_scan->width = 1;
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
		if (p.x < center.x + sensor_height / 2 && p.x > center.x - sensor_height / 2 &&
			p.y < center.y + sensor_width / 2 && p.y > center.y - sensor_width / 2)
		{
			cloud_projection->push_back(p);
		}
	}
}

int drawLaserImage(PointXYZRGB pin_hole, Mat* image_out, int sensor_pixel_height, int sensor_pixel_width, PointCloud<PointXYZRGB>::Ptr cloud_projection) {
	int image_point_added = 0;
	Mat image(sensor_pixel_height, sensor_pixel_width, CV_8UC3);
	float sensor_width = sensor_pixel_width * PIXEL_DIMENSION;
	float sensor_height = sensor_pixel_height * PIXEL_DIMENSION;
	float x_sensor_origin = pin_hole.x - (sensor_height) / 2;
	float y_sensor_origin = pin_hole.y - (sensor_width) / 2;

	// inizializza l'immagine
	for (int i = 0; i < sensor_pixel_height; i++)
		for (int j = 0; j < sensor_pixel_width; j++) {
			image.at<Vec3b>(i, j)[0] = 255;
			image.at<Vec3b>(i, j)[1] = 255;
			image.at<Vec3b>(i, j)[2] = 255;
	}

	for (int i = 0; i < cloud_projection->points.size(); i++)
	{
		float x, y;
		x = cloud_projection->points[i].x;
		y = cloud_projection->points[i].y;
		//cout << "(" << x << "," << y<< ") - ";

		int x_pos = ((x - x_sensor_origin) / PIXEL_DIMENSION);
		int y_pos = ((y - y_sensor_origin) / PIXEL_DIMENSION);
		//cout << "(" << x_pos << "," << y_pos <<")" << endl;

		if (x_pos >= 0 && x_pos < sensor_pixel_height && y_pos >= 0 && y_pos < sensor_pixel_width) {

			Vec3b & color = image.at<Vec3b>(x_pos, y_pos);
			color[0] = 0;
			color[1] = 0;
			color[2] = 0;
			image_point_added++;
		}
	}

	*image_out = image;

	return image_point_added;
}




int main(int argc, char** argv)
{
	// Load STL file as a PolygonMesh
	PolygonMesh mesh;
	PointCloud<PointXYZRGB>::Ptr cloud_projection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	Mat image;

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

	// Ordino gli array per una ricerca più efficiente dei poligoni
	float *tmp_a = new float[mesh.polygons.size()]; // li creo qui fuori perché creandoli ogni volta nella ricorsione
	int *tmp_b = new int[mesh.polygons.size()];     // c'è un crash dovuto alla ricorsione dell'operatore new
	mergesort(min_poligon_point, min_poligon_index, 0, mesh.polygons.size() - 1, tmp_a, tmp_b);
	delete[] tmp_a, tmp_b; // elimino gli array temporanei

	// Inizializza il laser
	initializeLaser(scanDirection);

	// Inizializza il Pin Hole
	initializePinHole(scanDirection, 650);

	// cerca i punti di insersezione del raggio laser
	findPointsMeshLaserIntersection(mesh, laser_point, 0.001, cloud_intersection, scanDirection);

	// effettua la proiezione dei punti di insersezione
	sensorPointProjection(focal_distance, sensor_height, sensor_width, cloud_intersection, cloud_projection);
	cout << "Proiezioni " << cloud_projection->points.size() << endl;
	cout << "Punti veri " << cloud_intersection->points.size() << endl;


	//****************** Converto la point cloud in un immagine **********************
	
	// crea immagine
	int image_point_added = drawLaserImage(pin_hole, &image, sensor_pixel_height, sensor_pixel_width, cloud_projection);
	cout << "Punti immagine aggiunti: " << image_point_added << endl;

	cloud_projection->push_back(pin_hole);

	// disegna contorni sensore
	drawSensor(pin_hole, focal_distance, sensor_width, sensor_height, cloud_projection);

	namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.
	imwrite("out.png", image);

	// Create a PCLVisualizer
	visualization::PCLVisualizer viewer("Mesh");
	viewer.addCoordinateSystem(100, "mesh");
	viewer.addPolygonMesh(mesh, "mesh");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb3(cloud_intersection);
	viewer.addPointCloud<PointXYZRGB>(cloud_intersection, rgb3, "cloudScan");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb4(cloud_projection);
	viewer.addPointCloud<PointXYZRGB>(cloud_projection, rgb4, "cloudProj");
	viewer.spin();


	return 0;
}


