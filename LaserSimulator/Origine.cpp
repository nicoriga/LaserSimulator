/*
* LaserSimulator
* Created on: 02/02/2016
* Last Update: 21/04/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
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


/// OpenCL parameter
#define RUN 256
#define LOCAL_SIZE 128
#define EPSILON 0.000001

#define DIRECTION_SCAN_AXIS_X 0
#define DIRECTION_SCAN_AXIS_Y 1
#define LASER_1 1
#define LASER_2 -1

#define X 0
#define Y 1
#define Z 2

Eigen::Matrix<double, 3, 1> typedef Vector3d;

struct Camera
{
	Mat camera_matrix;
	Mat distortion;
	int image_width;
	int image_height;
	float fps;
	float pixel_dimension;
};

struct Plane {
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
	Vec3 vertex_1;
	Vec3 vertex_2;
	Vec3 vertex_3;
};

struct MeshBounds
{
	float min_x = VTK_FLOAT_MAX;
	float min_y = VTK_FLOAT_MAX;
	float min_z = VTK_FLOAT_MAX;
	float max_x = VTK_FLOAT_MIN;
	float max_y = VTK_FLOAT_MIN;
	float max_z = VTK_FLOAT_MIN;
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

	vector<cl::Device> devices;
	vector<cl::Platform> platforms;
};

struct SimulationParams 
{
	float baseline;
	float height_to_mesh;
	float laser_aperture;
	float laser_inclination;
	float ray_density;
	float scan_speed;
	int scan_direction;
};

Vec3 calculateEdges(const Triangle &triangles) {

	float diff_x, diff_y, diff_z;
	Vec3 ret;

	diff_x = triangles.vertex_1.points[0] - triangles.vertex_2.points[0];
	diff_y = triangles.vertex_1.points[1] - triangles.vertex_2.points[1];
	diff_z = triangles.vertex_1.points[2] - triangles.vertex_2.points[2];

	ret.points[0] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	diff_x = triangles.vertex_1.points[0] - triangles.vertex_3.points[0];
	diff_y = triangles.vertex_1.points[1] - triangles.vertex_3.points[1];
	diff_z = triangles.vertex_1.points[2] - triangles.vertex_3.points[2];

	ret.points[1] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	diff_x = triangles.vertex_2.points[0] - triangles.vertex_3.points[0];
	diff_y = triangles.vertex_2.points[1] - triangles.vertex_3.points[1];
	diff_z = triangles.vertex_2.points[2] - triangles.vertex_3.points[2];

	ret.points[2] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	return ret;
};

void readParamsFromXML(Camera *camera, SimulationParams *params, bool *snapshot_save_flag, string *path_file)
	{
		// Read input parameters from xml file
		FileStorage fs("laser_simulator_params.xml", FileStorage::READ);
		if (fs.isOpened())
		{
			fs["path_file"] >> *path_file;
			fs["baseline"] >> params->baseline;
			fs["height_to_mesh"] >> params->height_to_mesh;
			fs["laser_aperture"] >> params->laser_aperture;
			fs["laser_inclination"] >> params->laser_inclination;
			fs["ray_density"] >> params->ray_density;
			fs["scan_speed"] >> params->scan_speed;
			fs["scan_direction"] >> params->scan_direction;
			fs["snapshot_save_flag"] >> *snapshot_save_flag;
			fs["camera_fps"] >> camera->fps;
			fs["image_width"] >> camera->image_width;
			fs["image_height"] >> camera->image_height;
			fs["pixel_dimension"] >> camera->pixel_dimension;
			fs["camera_matrix"] >> camera->camera_matrix;
			fs["camera_distortion"] >> camera->distortion;
		}
		else
		{
			cerr << "Error: cannot read the parameters" << endl;
			exit(-1);
		}

		/*
		*baseline = 600.f;					// [500, 800]
		*height_to_mesh = 1200.f;	
		*laser_aperture = 45.f;				// [30, 45]
		*laser_inclination = 60.f;			// [60, 70]
		*ray_density = 0.0015f;
		*scan_speed = 100.f;				// [100, 1000]


		*path_file = "../dataset/prodotto.stl";

		*scan_direction = DIRECTION_SCAN_AXIS_Y;

		*snapshot_save_flag = FALSE;

		// Intrinsic camera parameters
		camera->fps = 100.f;				// [100, 500]
		camera->image_width = 2024;
		camera->image_height = 1088;
		camera->pixel_dimension = 0.0055f; // mm

		camera->camera_matrix = Mat::zeros(3, 3, CV_64F);
		camera->camera_matrix.at<double>(0, 0) = 4615.04; // Fx
		camera->camera_matrix.at<double>(1, 1) = 4615.51; // Fy
		camera->camera_matrix.at<double>(0, 2) = 1113.41; // Cx
		camera->camera_matrix.at<double>(1, 2) = 480.016; // Cy
		camera->camera_matrix.at<double>(2, 2) = 1;

		camera->distortion = Mat::zeros(5, 1, CV_64F);
		camera->distortion.at<double>(0, 0) = -0.0506472;
		camera->distortion.at<double>(1, 0) = -1.45132;
		camera->distortion.at<double>(2, 0) = 0.000868025;
		camera->distortion.at<double>(3, 0) = 0.00298601;
		camera->distortion.at<double>(4, 0) = 8.92225;
		*/


		/*
		// Write file xml  
		// PER SALVARE COME YAML BASTA MODIFICARE IL NOME
		FileStorage fs("laser_simulator_params.xml", FileStorage::WRITE);
		fs << "path_file" << *path_file;
		fs << "baseline" << *baseline;					
		fs << "height_to_mesh" << *height_to_mesh;
		fs << "laser_aperture" << *laser_aperture;				
		fs << "laser_inclination" << *laser_inclination;
		fs << "ray_density" << *ray_density;
		fs << "scan_speed" << *scan_speed;
		fs << "scan_direction" << *scan_direction;
		fs << "snapshot_save_flag" << *snapshot_save_flag;
		fs << "camera_fps" << camera->fps;
		fs << "image_width" << camera->image_width;
		fs << "image_height" << camera->image_height;
		fs << "pixel_dimension" << camera->pixel_dimension;
		fs << "camera_matrix" << camera->camera_matrix;
		fs << "camera_distortion" << camera->distortion;
		fs.release();
		*/

		if (params->scan_speed < 100)
		{
			params->scan_speed = 100.f;
			cout << "WARNING: Velocita' di scansione inferiore a 100 (verra' impostata automaticamente a 100)" << endl << endl;
		}

		if (params->scan_speed > 1000)
		{
			params->scan_speed = 1000.f;
			cout << "WARNING: Velocita' di scansione superiore a 1000 (verra' impostata automaticamente a 1000)" << endl << endl;
		}	
		
		if (params->baseline < 500)
		{
			params->baseline = 500.f;
			cout << "WARNING: Baseline inferiore a 500 (verra' impostata automaticamente a 500)" << endl << endl;
		}

		if (params->baseline > 800)
		{
			params->baseline = 800.f;
			cout << "WARNING: Baseline superiore a 800 (verra' impostata automaticamente a 800)" << endl << endl;
		}

		if (camera->fps < 100)
		{
			camera->fps = 100.f;
			cout << "WARNING: FPS della camera inferiori a 100 (verranno impostati automaticamente a 100)" << endl << endl;
		}

		if (camera->fps > 500)
		{
			camera->fps = 500.f;
			cout << "WARNING: FPS della camera superiori a 500 (verranno impostati automaticamente a 500)" << endl << endl;
		}
	}

void arraysMerge(float *a, int *b, int low, int high, int mid, float *c, int *d)
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

	for (i = low; i < k; ++i)
	{
		a[i] = c[i];
		b[i] = d[i];
	}
}

void arraysMergesort(float *a, int* b, int low, int high, float *tmp_a, int *tmp_b)
{
	int mid;

	if (low < high)
	{
		mid = (low + high) / 2;
		arraysMergesort(a, b, low, mid, tmp_a, tmp_b);
		arraysMergesort(a, b, mid + 1, high, tmp_a, tmp_b);
		arraysMerge(a, b, low, high, mid, tmp_a, tmp_b);
	}
	return;
}

void updateMinMax(PointXYZRGB point, MeshBounds *bounds) {
	if (point.x < bounds->min_x)
		bounds->min_x = point.x;
	if (point.x > bounds->max_x)
		bounds->max_x = point.x;

	if (point.y < bounds->min_y)
		bounds->min_y = point.y;
	if (point.y > bounds->max_y)
		bounds->max_y = point.y;

	if (point.z < bounds->min_z)
		bounds->min_z = point.z;
	if (point.z > bounds->max_z)
		bounds->max_z = point.z;
}

void calculateBoundariesAndArrayMax(const SimulationParams &params, PolygonMesh mesh, int* max_point_triangle_index, float* max_point_triangle, MeshBounds *bounds) {

	PointCloud<PointXYZ> cloud_mesh;
	PointXYZRGB point_1, point_2, point_3;

	// Convert mesh in a point cloud 
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// ricerca max e min per tutti gli assi
	for (int i = 0; i < mesh.polygons.size(); i++)
	{
		point_1.x = cloud_mesh.points[mesh.polygons[i].vertices[0]].x;
		point_1.y = cloud_mesh.points[mesh.polygons[i].vertices[0]].y;
		point_1.z = cloud_mesh.points[mesh.polygons[i].vertices[0]].z;

		updateMinMax(point_1, bounds);

		point_2.x = cloud_mesh.points[mesh.polygons[i].vertices[1]].x;
		point_2.y = cloud_mesh.points[mesh.polygons[i].vertices[1]].y;
		point_2.z = cloud_mesh.points[mesh.polygons[i].vertices[1]].z;

		updateMinMax(point_2, bounds);

		point_3.x = cloud_mesh.points[mesh.polygons[i].vertices[2]].x;
		point_3.y = cloud_mesh.points[mesh.polygons[i].vertices[2]].y;
		point_3.z = cloud_mesh.points[mesh.polygons[i].vertices[2]].z;

		updateMinMax(point_3, bounds);

		// Popolamento array max_point_triangle per tener traccia quale dei 3 vertici ha la Y più piccola
		max_point_triangle_index[i] = i;

		if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
		{
			if (point_1.x > point_2.x && point_1.x > point_3.x)
				max_point_triangle[i] = point_1.x;
			else
			{
				if (point_2.x > point_3.x)
					max_point_triangle[i] = point_2.x;
				else
					max_point_triangle[i] = point_3.x;
			}
		}

		if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
		{
			if (point_1.y > point_2.y && point_1.y > point_3.y)
				max_point_triangle[i] = point_1.y;
			else
			{
				if (point_2.y > point_3.y)
					max_point_triangle[i] = point_2.y;
				else
					max_point_triangle[i] = point_3.y;
			}
		}
	}
}

void setInitialPosition(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, const SimulationParams &params,
	float inclination_coefficient, const MeshBounds &bounds) {
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		laser_origin_1->z = bounds.max_z + params.height_to_mesh;
		laser_origin_1->x = (bounds.max_x + bounds.min_x) / 2;
		laser_origin_1->y = bounds.min_y - (laser_origin_1->z - bounds.min_z) * inclination_coefficient;

		laser_origin_2->z = laser_origin_1->z;
		laser_origin_2->x = laser_origin_1->x;
		laser_origin_2->y = laser_origin_1->y + 2 * params.baseline;

		pin_hole->x = laser_origin_1->x;
		pin_hole->y = laser_origin_1->y + params.baseline;
		pin_hole->z = laser_origin_1->z;
	}

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		laser_origin_1->z = bounds.max_z + params.height_to_mesh;
		laser_origin_1->y = (bounds.max_y + bounds.min_y) / 2;
		laser_origin_1->x = bounds.min_x - (laser_origin_1->z - bounds.min_z) * inclination_coefficient;

		laser_origin_2->z = laser_origin_1->z;
		laser_origin_2->y = laser_origin_1->y;
		laser_origin_2->x = laser_origin_1->x + 2 * params.baseline;

		pin_hole->x = laser_origin_1->x + params.baseline;
		pin_hole->y = laser_origin_1->y;
		pin_hole->z = laser_origin_1->z;
	}
}

void setLasersAndPinHole(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, float current_position, const SimulationParams &params) {
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		pin_hole->y = current_position;
		pin_hole->x = laser_origin_1->x;

		laser_origin_1->y = pin_hole->y - params.baseline;
		laser_origin_2->y = pin_hole->y + params.baseline;

	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		pin_hole->x = current_position;
		pin_hole->y = laser_origin_1->y;

		laser_origin_1->x = pin_hole->x - params.baseline;
		laser_origin_2->x = pin_hole->x + params.baseline;

	}
	pin_hole->z = laser_origin_1->z;
}

Vec3f directionVector(PointXYZRGB source_point, PointXYZRGB destination_point) {
	Vec3f direction;

	direction[0] = (destination_point.x - source_point.x) / (destination_point.z - source_point.z);
	direction[1] = (destination_point.y - source_point.y) / (destination_point.z - source_point.z);
	direction[2] = 1;

	return direction;
}

// Moller-Trumbore intersection algorithm
int triangleIntersection(const Vector3d V1, const Vector3d V2, const Vector3d V3,
	const Vector3d O, const Vector3d D, float* out, Vector3d &intPoint)
{
	Vector3d e1, e2;  //Edge1, Edge2
	Vector3d P, Q, T;
	float det, inv_det, u, v;
	float t;

	//Find vectors for two edges sharing V1
	e1 = V2 - V1;
	e2 = V3 - V1;

	//Begin calculating determinant - also used to calculate u parameter
	P = D.cross(e2);
	//if determinant is near zero, ray lies in plane of triangle
	det = e1.dot(P);

	//NOT CULLING
	if (det > -EPSILON && det < EPSILON)
		return 0;
	inv_det = 1.f / det;

	//calculate distance from V1 to ray origin
	T = O - V1;

	//Calculate u parameter and test bound
	u = T.dot(P) * inv_det;
	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f)
		return 0;

	//Prepare to test v parameter
	Q = T.cross(e1);

	//Calculate V parameter and test bound
	v = D.dot(Q) * inv_det;
	//The intersection lies outside of the triangle
	if (v < 0.f || u + v  > 1.f)
		return 0;

	//t = DOT(e2, Q) * inv_det;
	t = e2.dot(Q) * inv_det;

	if (t > EPSILON) { //ray intersection
		*out = t;

		intPoint = O + t*D;

		return 1;
	}

	return 0;
}

// Ritorna la coordinata della direzione di scansione in cui interseca
float rayPlaneLimitIntersection(const PointXYZ &start_point, const Vector3d &direction, float plane_coordinate, int scan_direction) {
	if (scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		return direction[1] * (plane_coordinate - start_point.z) / direction[2] + start_point.y;
	}
	if (scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		return direction[0] * (plane_coordinate - start_point.z) / direction[2] + start_point.x;
	}

	return 0;
}

void getPlaneCoefficents(const PointXYZ &laser, const Vector3d &line_1, const Vector3d &line_2, Plane* plane) {
	Vector3d plane_normal = line_1.cross(line_2);
	plane->A = plane_normal[0];
	plane->B = plane_normal[1];
	plane->C = plane_normal[2];
	plane->D = -plane_normal[0] * laser.x - plane_normal[1] * laser.y - plane_normal[2] * laser.z;
}

int getLowerBound(float* array_points, int array_size, float threshold) {
	int index = array_size - 1;

	for (int i = 0; i < array_size; i++)
	{
		if (array_points[i] > threshold) {
			index = i - 1;
			break;
		}
	}
	if (index < 0)
		index = 0;

	return index;
}

int getUpperBound(float* array_points, int array_size, float threshold) {
	int index = 0;

	for (int i = array_size - 1; i > 0; i--)
	{
		if (array_points[i] < threshold) {
			index = i;
			break;
		}
	}

	return index;
}

/*void findPointsMeshLaserIntersection(const PolygonMesh mesh, const PointXYZ laser, const float density, PointCloud<PointXYZRGB>::Ptr cloudIntersection, int scanDirection, 
										Plane* plane, float aperture_coefficientURE, float inclination_coefficientINATION, int* max_point_triangle_index,
										float* max_point_triangle, double laser_number, float min_z, float max_z)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	const float MIN_INTERSECTION = VTK_FLOAT_MIN;

	int number_of_line = (aperture_coefficientURE * 2) / density;
	//cout << "Numero linee fascio laser: " << number_of_line << endl;

	int d1, d2;
	if (scanDirection == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
		Vector3d line_1(-aperture_coefficientURE + 0 * density, laser_number * inclination_coefficientINATION, -1);
		Vector3d line_2(-aperture_coefficientURE + 10 * density, laser_number * inclination_coefficientINATION, -1);
		getPlaneCoefficents(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(0, -tan(deg2rad(laser_aperture / 2)) + 0 * density, -1), 1000);

	}
	if (scanDirection == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
		Vector3d line_1(laser_number * inclination_coefficientINATION, -aperture_coefficientURE + 0 * density, -1);
		Vector3d line_2(laser_number * inclination_coefficientINATION, -aperture_coefficientURE + 1000 * density, -1);

		getPlaneCoefficents(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(-tan(deg2rad(laser_aperture / 2)) + 0 * density , 0, -1), 1000);

	}

	Vector3d direction_ray_start;
	direction_ray_start[d1] = -aperture_coefficientURE;
	direction_ray_start[d2] = laser_number * inclination_coefficientINATION;
	direction_ray_start[2] = -1;
	float laser_intersect_min_z = rayPlaneLimitIntersection(laser, direction_ray_start, min_z, scanDirection);
	float laser_intersect_max_z = rayPlaneLimitIntersection(laser, direction_ray_start, max_z, scanDirection);

	int lower_bound, upper_bound;

	if (laser_number == LASER_2)
	{
		lower_bound = getLowerBound(max_point_triangle, mesh.polygons.size(), laser_intersect_min_z);
		upper_bound = getUpperBound(max_point_triangle, mesh.polygons.size(), laser_intersect_max_z);
	}

	else //LASER_1
	{
		lower_bound = getUpperBound(max_point_triangle, mesh.polygons.size(), laser_intersect_max_z);
		upper_bound = getLowerBound(max_point_triangle, mesh.polygons.size(), laser_intersect_min_z);
	}

	//cout << "laser_intersect_min_z: " << laser_intersect_min_z << endl;
	//cout << "laser_intersect_max_z: " << laser_intersect_max_z << endl;
	//cout << "start_index: " << start_index << endl;
	//cout << "final_index: " << final_index << endl;
	cout << "Number of Poligon intersected: " << upper_bound - lower_bound << endl;

#pragma omp parallel for //ordered schedule(dynamic)
	for (int j = 0; j < number_of_line; j++)
	{
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();

		PointXYZ tmp;
		Vertices triangle;
		Vector3d vertex_1, vertex_2, vertex_3;
		Vector3d intersection_point, origin_ray, direction_ray;
		float out;
		PointXYZRGB firstIntersection;

		origin_ray[0] = laser.x;
		origin_ray[1] = laser.y;
		origin_ray[2] = laser.z;

		float i = -aperture_coefficientURE + j*density;

		firstIntersection.z = MIN_INTERSECTION;

		direction_ray[d1] = i;
		direction_ray[d2] = laser_number * inclination_coefficientINATION;
		//direction_ray[2] = -1;

		for (int k = lower_bound; k < upper_bound; k++)
		{
			//triangle = mesh.polygons.at(max_point_triangle_index[k]);
			tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[0]];
			vertex_1[0] = tmp.x;
			vertex_1[1] = tmp.y;
			vertex_1[2] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[1]];
			vertex_2[0] = tmp.x;
			vertex_2[1] = tmp.y;
			vertex_2[2] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[2]];
			vertex_3[0] = tmp.x;
			vertex_3[1] = tmp.y;
			vertex_3[2] = tmp.z;

			if (triangleIntersection(vertex_1, vertex_2, vertex_3, origin_ray, direction_ray, &out, intersection_point) != 0)
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
}*/

void prepareDataForOpenCL(const PolygonMesh &mesh, Triangle* triangles, int* max_point_triangle_index) {
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	PointXYZ tmp;


	for (int k = 0; k < mesh.polygons.size(); k++)
	{

		tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[0]];
		triangles[k].vertex_1.points[X] = tmp.x;
		triangles[k].vertex_1.points[Y] = tmp.y;
		triangles[k].vertex_1.points[Z] = tmp.z;

		tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[1]];
		triangles[k].vertex_2.points[X] = tmp.x;
		triangles[k].vertex_2.points[Y] = tmp.y;
		triangles[k].vertex_2.points[Z] = tmp.z;

		tmp = meshVertices.points[mesh.polygons[max_point_triangle_index[k]].vertices[2]];
		triangles[k].vertex_3.points[X] = tmp.x;
		triangles[k].vertex_3.points[Y] = tmp.y;
		triangles[k].vertex_3.points[Z] = tmp.z;
	}
}

int initializeOpenCL(OpenCLDATA* openCLData, Triangle* triangle_array, int array_lenght, int array_size_hits) {

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
		//openCLData->context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
		openCLData->context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
		openCLData->devices = openCLData->context.getInfo<CL_CONTEXT_DEVICES>();

		// Create command queue for first device
		openCLData->queue = cl::CommandQueue(openCLData->context, openCLData->devices[0], 0, &err);

		FILE* programHandle;
		size_t kernelSourceSize;
		char *kernelSource;

		// get size of kernel source
		programHandle = fopen("IntersectionTriangle.cl", "rb");
		fseek(programHandle, 0, SEEK_END);
		kernelSourceSize = ftell(programHandle);
		rewind(programHandle);

		// read kernel source into buffer
		kernelSource = (char*)malloc(kernelSourceSize + 1);
		kernelSource[kernelSourceSize] = '\0';
		fread(kernelSource, sizeof(char), kernelSourceSize, programHandle);
		fclose(programHandle);

		//Build kernel from source string
		openCLData->program_ = cl::Program(openCLData->context, kernelSource);
		err = openCLData->program_.build(openCLData->devices);

		free(kernelSource);

		// Size, in bytes, of each vector
		openCLData->triangles_size = array_lenght*sizeof(Triangle);
		openCLData->points_size = array_size_hits*sizeof(Vec3);
		openCLData->hits_size = array_size_hits*sizeof(uchar);

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

int computeOpenCL(OpenCLDATA* openCLData, Vec3* output_points, uchar* output_hits, int start_index, int array_lenght, const Vec3 &ray_origin, const Vec3 &ray_direction) {

	//high_resolution_clock::time_point start;
	//start = high_resolution_clock::now();

	cl_int err = CL_SUCCESS;

	openCLData->kernel.setArg(3, start_index);
	openCLData->kernel.setArg(4, array_lenght);
	openCLData->kernel.setArg(5, ray_origin);
	openCLData->kernel.setArg(6, ray_direction);

	// Number of work items in each local work group

	cl::NDRange localSize(LOCAL_SIZE, 1, 1);
	// Number of total work items - localSize must be devisor
	int global_size = (int)(ceil((array_lenght / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
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

void findPointsMeshLaserIntersectionOpenCL(OpenCLDATA* openCLData, Triangle* all_triangles, const vector<Triangle> &big_triangles, Vec3* output_points, uchar* output_hits,
	const PolygonMesh &mesh, const PointXYZ &laser, const SimulationParams &params, PointCloud<PointXYZRGB>::Ptr cloud_intersection, Plane* plane, 
	float aperture_coefficient, float inclination_coefficient, float* max_point_triangle, const int laser_number, const MeshBounds &bounds)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	int array_size_hits = (int) (ceil(mesh.polygons.size() / (float)RUN));
	
	int number_of_line = (aperture_coefficient * 2) / params.ray_density;

	int d1, d2;
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
		Vector3d line_1(-aperture_coefficient + 0 * params.ray_density, laser_number * inclination_coefficient, -1);
		Vector3d line_2(-aperture_coefficient + 10 * params.ray_density, laser_number * inclination_coefficient, -1);
		getPlaneCoefficents(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(-aperture_coefficientURE, inclination_coefficientINATION, -1), 1500);
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
		Vector3d line_1(laser_number * inclination_coefficient, -aperture_coefficient + 0 * params.ray_density, -1);
		Vector3d line_2(laser_number * inclination_coefficient, -aperture_coefficient + 10 * params.ray_density, -1);
		getPlaneCoefficents(laser, line_1, line_2, plane);

		//drawLine(cloudIntersection, laser, Eigen::Vector3f(-tan(deg2rad(laser_aperture / 2)) + 0 * density , 0, -1), 1000);
	}

	Vector3d direction_ray_start;
	direction_ray_start[d1] = -aperture_coefficient;
	direction_ray_start[d2] = laser_number * inclination_coefficient;
	direction_ray_start[2] = -1;
	float laser_intersect_min_z = rayPlaneLimitIntersection(laser, direction_ray_start, bounds.min_z, params.scan_direction);
	float laser_intersect_max_z = rayPlaneLimitIntersection(laser, direction_ray_start, bounds.max_z, params.scan_direction);

	int lower_bound, upper_bound;

	// Bounds calculated due to laser
	switch (laser_number)
	{
		case (LASER_1):
			lower_bound = getLowerBound(max_point_triangle, mesh.polygons.size(), laser_intersect_max_z);
			upper_bound = getUpperBound(max_point_triangle, mesh.polygons.size(), laser_intersect_min_z);
			break;

		case (LASER_2):
			lower_bound = getLowerBound(max_point_triangle, mesh.polygons.size(), laser_intersect_min_z);
			upper_bound = getUpperBound(max_point_triangle, mesh.polygons.size(), laser_intersect_max_z);
			break;
	}


	int diff = upper_bound - lower_bound;

	for (int j = 0; j < number_of_line; j++)
	{
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();

		PointXYZ tmp;
		Vertices triangle;
		Vector3d vertex_1, vertex_2, vertex_3;
		Vector3d intersection_point;
		Vec3 ray_origin, ray_direction;
		float out;
		PointXYZRGB firstIntersection;

		ray_origin.points[X] = laser.x;
		ray_origin.points[Y] = laser.y;
		ray_origin.points[Z] = laser.z;

		float i = -aperture_coefficient + j * params.ray_density;

		firstIntersection.z = VTK_FLOAT_MIN;

		ray_direction.points[d1] = i;
		ray_direction.points[d2] = laser_number * inclination_coefficient;
		ray_direction.points[Z] = -1;

		//Triangle* triangles = all_triangles + start_index;
		//Vec3* output_points = new Vec3[diff];
		//int* output_hits = new int[diff];

		if (diff > 0)
		{
			computeOpenCL(openCLData, output_points, output_hits, lower_bound, diff, ray_origin, ray_direction);

			int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
			for (int h = 0; h < n_max; h++)
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
		}

		Vector3d origin_ray(ray_origin.points[X], ray_origin.points[Y], ray_origin.points[Z]);
		Vector3d direction_ray(ray_direction.points[d1], ray_direction.points[d2], ray_direction.points[Z]);

		for (int k = 0; k < big_triangles.size(); k++)
		{
			//triangle = mesh.polygons.at(max_point_triangle_index[k]);
			vertex_1[0] = big_triangles[k].vertex_1.points[X];
			vertex_1[1] = big_triangles[k].vertex_1.points[Y];
			vertex_1[2] = big_triangles[k].vertex_1.points[Z];

			vertex_2[0] = big_triangles[k].vertex_2.points[X];
			vertex_2[1] = big_triangles[k].vertex_2.points[Y];
			vertex_2[2] = big_triangles[k].vertex_2.points[Z];

			vertex_3[0] = big_triangles[k].vertex_3.points[X];
			vertex_3[1] = big_triangles[k].vertex_3.points[Y];
			vertex_3[2] = big_triangles[k].vertex_3.points[Z];

			if (triangleIntersection(vertex_1, vertex_2, vertex_3, origin_ray, direction_ray, &out, intersection_point) != 0)
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


		if (firstIntersection.z > VTK_FLOAT_MIN)
			cloud_intersection->push_back(firstIntersection);

		//duration<double> timer3 = high_resolution_clock::now() - start;
		//cout << "Total time cycle ray intersection OpenCL:" << timer3.count() * 1000 << endl;
	}
}

bool checkOcclusion(const PointXYZRGB &point, const PointXYZ &pin_hole, float* max_point_triangle, int polygon_size, OpenCLDATA* openCLData, Triangle* all_triangles,
	Vec3* output_points, uchar* output_hits) {
	/*

	1. calcola il raggio tra il point e il pin_hole
	2. trova gli indici nell'array dei min_y tra le coordinate y del pin_hole e del point
	3. cerco intersezione tra il raggio e i triangoli
	5. interseca con un triangolo?
	Falso -> return 0
	return 1

	*/

	Vec3 origin;
	origin.points[X] = point.x;
	origin.points[Y] = point.y;
	origin.points[Z] = point.z;

	Vec3 direction;
	direction.points[X] = pin_hole.x - point.x;
	direction.points[Y] = pin_hole.y - point.y;
	direction.points[Z] = pin_hole.z - point.z;

	int lower_bound, upper_bound;

	if (pin_hole.y < origin.points[Y])
	{
		lower_bound = getLowerBound(max_point_triangle, polygon_size, pin_hole.y);
		upper_bound = getUpperBound(max_point_triangle, polygon_size, origin.points[Y]);
	}
	else
	{
		lower_bound = getLowerBound(max_point_triangle, polygon_size, origin.points[Y]);
		upper_bound = getUpperBound(max_point_triangle, polygon_size, pin_hole.y);
	}

	int diff = upper_bound - lower_bound;

	if (diff > 0)
	{
		computeOpenCL(openCLData, output_points, output_hits, lower_bound, diff, origin, direction);

		int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
		for (int k = 0; k < n_max; k++)
		{
			if (output_hits[k] == 1)
				return FALSE;
		}

	}

	return TRUE;
}

void cameraSnapshot(const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_1, const PointXYZ &laser_2, PointCloud<PointXYZRGB>::Ptr cloudIntersection,
					Mat* img, const SimulationParams &params, int polygon_size, OpenCLDATA* openCLData, Triangle* all_triangles, Vec3* output_points,
					uchar* output_hits, float* max_point_triangle) {
	// Initialize a white image
	Mat image(camera.image_height, camera.image_width, CV_8UC3, Scalar(255, 255, 255));

	PointCloud<PointXYZ>::Ptr cloud_src(new PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr cloud_target(new PointCloud<PointXYZ>);
	PointXYZ current_point;

	cloud_src->push_back(pin_hole);
	cloud_src->push_back(laser_1);
	cloud_src->push_back(laser_2);
	
	PointXYZ c, p1, p2;
	// camera
	c.x = 0;
	c.y = 0;
	c.z = 0;
	cloud_target->push_back(c);

	// laser
	//if (scan_direction == DIRECTION_SCAN_AXIS_Y)
	//{
		p1.x = 0;
		p1.y = -params.baseline;
		p1.z = 0;

		p2.x = 0;
		p2.y = params.baseline;
		p2.z = 0;
	//}
	/*if (scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		p1.x = 0;
		p1.y = -baseline;
		p1.z = 0;

		p2.x = 0;
		p2.y = baseline;
		p2.z = 0;
	}*/
	cloud_target->push_back(p1);
	cloud_target->push_back(p2);

	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>  transEst;
	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>::Matrix4 trans;
	transEst.estimateRigidTransformation(*cloud_src, *cloud_target, trans);

	std::vector<Point3d> points;
	std::vector<Point2d> output_point;

	for (int i = 0; i < cloudIntersection->size(); i++) {
		Eigen::Vector4f v_point, v_point_final;
		v_point[0] = cloudIntersection->points[i].x;
		v_point[1] = cloudIntersection->points[i].y;
		v_point[2] = cloudIntersection->points[i].z;
		v_point[3] = 1;
		v_point_final = trans * v_point;

		v_point_final[2] = -v_point_final[2];

		Point3f p(v_point_final[0], v_point_final[1], v_point_final[2]);

		points.push_back(p);
	}

	if (cloudIntersection->size() > 0) {
		projectPoints(points, Mat::zeros(3, 1, CV_64F), Mat::zeros(3, 1, CV_64F), camera.camera_matrix, camera.distortion, output_point);
		Point2d pixel;
		for (int i = 0; i < output_point.size(); i++) {
			pixel = output_point.at(i);
			pixel.x += 0.5;
			pixel.y += 0.5;

			if ((pixel.y >= 0) && (pixel.y < image.rows) && (pixel.x >= 0) && (pixel.x < image.cols))
			{
				if (checkOcclusion(cloudIntersection->at(i), pin_hole, max_point_triangle, polygon_size, openCLData, all_triangles, output_points, output_hits))
				{
					image.at<Vec3b>((int)(pixel.y), (int)(pixel.x))[0] = 0;
					image.at<Vec3b>((int)(pixel.y), (int)(pixel.x))[1] = 0;
					image.at<Vec3b>((int)(pixel.y), (int)(pixel.x))[2] = 0;
				}
			}
		}

	}

	*img = image;
}


void imageToCloud(Camera &camera, const SimulationParams &params, const Plane &plane_1, const Plane &plane_2, const PointXYZ &pin_hole, Mat* image, int roi1_start, int roi2_start, int roi_dimension,
					PointCloud<PointXYZ>::Ptr cloud_out) {
	PointXYZ point;    // The point to add at the cloud
	float dx, dy, dz;  // Directional vector for the line pin_hole - point in the sensor
	float x_sensor_origin, y_sensor_origin; // Origin of the sensor in the space

	// Amount of traslate of the sensor compared to the pinhole
	float delta_x = ((image->cols / 2) - camera.camera_matrix.at<double>(0, 2)) * camera.pixel_dimension;
	float delta_y = ((image->rows / 2) - camera.camera_matrix.at<double>(1, 2)) * camera.pixel_dimension;
	
	// Computation of the focal_length
	float focal_length_x = camera.camera_matrix.at<double>(0, 0) * camera.pixel_dimension;
	float focal_length_y = camera.camera_matrix.at<double>(1, 1) * camera.pixel_dimension;
	float focal_length = (focal_length_x + focal_length_y) / 2;

	// Traslation of the sensor
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		x_sensor_origin = pin_hole.x - (image->rows * camera.pixel_dimension) / 2 - delta_y;
		y_sensor_origin = pin_hole.y - (image->cols * camera.pixel_dimension) / 2 + delta_x;
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		x_sensor_origin = pin_hole.x + (image->cols * camera.pixel_dimension) / 2 - delta_x;
		y_sensor_origin = pin_hole.y - (image->rows * camera.pixel_dimension) / 2 - delta_y;
	}

	// Undistort the image accord with the camera disortion params
	Mat image_undistort;
	undistort(*image, image_undistort, camera.camera_matrix, camera.distortion);
	flip(image_undistort, *image, 0);

	// Project the ROI1 points on the first plane
	for (int j = 0; j < image->cols; j++)
	{
		for (int i = roi1_start; i < roi1_start + roi_dimension; i++)
		{
			Vec3b & color = image->at<Vec3b>(i, j);
			// Check the color of the pixels
			if (color[0] !=255 && color[1] != 255 && color[2] != 255) {
				// Put the points of the image in the virtual sensor in the space
				if (params.scan_direction == DIRECTION_SCAN_AXIS_X) {
					point.x = i * camera.pixel_dimension + x_sensor_origin;
					point.y = j * camera.pixel_dimension + y_sensor_origin;
				}
				if (params.scan_direction == DIRECTION_SCAN_AXIS_Y) {
					point.x = x_sensor_origin - j * camera.pixel_dimension;
					point.y = i * camera.pixel_dimension + y_sensor_origin;
				}
				point.z = pin_hole.z + focal_length;

				dx = pin_hole.x - point.x;
				dy = pin_hole.y - point.y;
				dz = pin_hole.z - point.z;

				// Project the point in the sensor on the laser plane passing by the pin hole
				float t = -(plane_1.A * point.x + plane_1.B * point.y + plane_1.C * point.z + plane_1.D) / (plane_1.A * dx + plane_1.B * dy + plane_1.C * dz);
				point.x = dx * t + point.x;
				point.y = dy * t + point.y;
				point.z = dz * t + point.z;
				cloud_out->push_back(point);

				break;
			}
		}
	}

	// Project the ROI2 points on the second plane
	for (int j = 0; j < image->cols; j++)
	{
		for (int i = roi2_start; i < roi2_start + roi_dimension; i++)
		{
			Vec3b & color = image->at<Vec3b>(i, j);
			// Check the color of the pixels
			if (color[0] != 255 && color[1] != 255 && color[2] != 255) {
				// Put the points of the image in the virtual sensor in the space
				if (params.scan_direction == DIRECTION_SCAN_AXIS_X) {
					point.x = i * camera.pixel_dimension + x_sensor_origin;
					point.y = j * camera.pixel_dimension + y_sensor_origin;
				}
				if (params.scan_direction == DIRECTION_SCAN_AXIS_Y) {
					point.x = x_sensor_origin - j * camera.pixel_dimension;
					point.y = i * camera.pixel_dimension + y_sensor_origin;
				}
				point.z = pin_hole.z + focal_length;

				dx = pin_hole.x - point.x;
				dy = pin_hole.y - point.y;
				dz = pin_hole.z - point.z;

				// Project the point in the sensor on the laser plane passing by the pin hole
				float t = -(plane_2.A * point.x + plane_2.B * point.y + plane_2.C * point.z + plane_2.D) / (plane_2.A * dx + plane_2.B * dy + plane_2.C * dz);
				point.x = dx * t + point.x;
				point.y = dy * t + point.y;
				point.z = dz * t + point.z;
				cloud_out->push_back(point);

				break;
			}
		}
	}
}

/*void drawLine(PointCloud<PointXYZRGB>::Ptr cloud, PointXYZ start_point, Eigen::Vector3f direction, int number_of_point) {
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
}*/

void printProgBar(int percent) {
	string bar;

	for (int i = 0; i < 50; i++) {
		if (i < (percent / 2)) {
			bar.replace(i, 1, "=");
		}
		else if (i == (percent / 2)) {
			bar.replace(i, 1, ">");
		}
		else {
			bar.replace(i, 1, " ");
		}
	}

	cout << "\r" "[" << bar << "] ";
	cout.width(3);
	cout << percent << "%     " << flush;
}

string returnTime(duration<double> timer)
{
	int sec = (int) timer.count() % 60;
	string seconds = sec < 10 ? "0" + to_string(sec) : to_string(sec);
	string minutes = to_string((int) (((int) timer.count() / 60) % 60));

	return minutes + ":" + seconds + " s";
}


int main(int argc, char** argv)
{
	PolygonMesh mesh;
	MeshBounds bounds;
	Camera camera;
	SimulationParams params;
	bool snapshot_save_flag;
	string path_file;
	OpenCLDATA openCLData;

	Mat image;
	Triangle* all_triangles;
	PointXYZ laser_origin_1, laser_origin_2, pin_hole;

	
	//********* Read data from XML parameters file ***************************************
	readParamsFromXML(&camera, &params, &snapshot_save_flag, &path_file);

	// Calculate inclination and angular coefficients
	float inclination_coefficient = tan(deg2rad(90.f - params.laser_inclination));
	float aperture_coefficient = tan(deg2rad(params.laser_aperture / 2.f));

	// Starting time counter
	high_resolution_clock::time_point start = high_resolution_clock::now();


	//********* Load STL file as a PolygonMesh *******************************************
	if (io::loadPolygonFileSTL(path_file, mesh) == 0)
	{
		PCL_ERROR("Failed to load STL file\n");
		return 1;
	}
	cout << "Dimensione della mesh (n. triangoli): " << mesh.polygons.size() << endl << endl;
	
	// Arrays used to optimize the computation of intersections
	float *max_point_triangle = new float[mesh.polygons.size()];
	int *max_point_triangle_index = new int[mesh.polygons.size()];
	

	//********** Find minimum and mixiumum points of 3 axis and fill *********************
	//********** arrays used to find maximum value on the direction axis *****************
	calculateBoundariesAndArrayMax(params, mesh, max_point_triangle_index, max_point_triangle, &bounds);


	//********** Print minimum and maximum points of mesh ********************************
	cout << "Estremi della mesh:" << endl;
	cout << "X: [" << bounds.min_x << ", " << bounds.max_x << "]        ";
	cout << "Y: [" << bounds.min_y << ", " << bounds.max_y << "]        ";
	cout << "Z: [" << bounds.min_z << ", " << bounds.max_z << "]" << endl << endl;


	//********** Sort arrays to have more efficency in the search ************************
	float *tmp_a = new float[mesh.polygons.size()];
	int *tmp_b = new int[mesh.polygons.size()];
	arraysMergesort(max_point_triangle, max_point_triangle_index, 0, mesh.polygons.size() - 1, tmp_a, tmp_b);
	delete[] tmp_a, tmp_b;
	
	//************************ OpenCL Loading ********************************************
	int array_size_hits = (int)(ceil(mesh.polygons.size() / (float)RUN));
	int size_array = mesh.polygons.size();
	all_triangles = new Triangle[size_array];
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	prepareDataForOpenCL(mesh, all_triangles, max_point_triangle_index);
	initializeOpenCL(&openCLData, all_triangles, size_array, array_size_hits);


	//************************ Find "big" triangles **************************************
	// Questa parte dovra' sparire e finire in opencl, a quel punto faremo una funzione opportuna
	vector<Triangle> big_triangles_vec;
	Vec3 coord;
	float projection_distance = (bounds.max_z - bounds.min_z) * inclination_coefficient;

	for (int i = 0; i < size_array; i++) {
		coord = calculateEdges(all_triangles[i]);
		if (coord.points[0] > projection_distance || coord.points[1] > projection_distance || coord.points[2] > projection_distance)
			big_triangles_vec.push_back(all_triangles[i]);
	}

	Triangle* big_triangles = new Triangle[big_triangles_vec.size()];
	for (int i = 0; i < big_triangles_vec.size(); i++)
		big_triangles[i] = big_triangles_vec[i];

	cout << "NUMERO BIG TRIANGLES: " << big_triangles_vec.size() << endl << endl;


	//**************** Initialize initial position for camera and lasers *****************
	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params,	inclination_coefficient, bounds);

	// Step size (in mm) between two snapshots
	float increment = params.scan_speed / camera.fps;


	float current_position, number_of_iterations, final_pos;

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		current_position = pin_hole.x;
		final_pos = bounds.max_x - (bounds.min_x - laser_origin_2.x);
		number_of_iterations = (final_pos - laser_origin_1.x) / increment;
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		current_position = pin_hole.y;
		final_pos = bounds.max_y + (bounds.min_y - laser_origin_2.y);
		number_of_iterations = (final_pos - laser_origin_1.y) / increment;
	}
	

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection_backup(new PointCloud<PointXYZRGB>);
	Plane plane_1, plane_2;

	// Disegno i laser
	/*if (scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		drawLine(cloud_intersection_backup, laser_origin_1, Eigen::Vector3f(-inclination_coefficient, -0, -1), 2000);
		drawLine(cloud_intersection_backup, laser_origin_2, Eigen::Vector3f(inclination_coefficient, 0, -1), 2000);
	}
	if (scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		drawLine(cloud_intersection_backup, laser_origin_1, Eigen::Vector3f(0, -inclination_coefficient, -1), 2000);
		drawLine(cloud_intersection_backup, laser_origin_2, Eigen::Vector3f(0, inclination_coefficient, -1), 2000);
	}*/



	//************CORE OF THE PROJECT: this cycle simulates the laser scan. **************
	//*********** Every iteration finds intersection with mesh, take a camera snapshot ***
	//*********** and reconstruct the points in the 3D space *****************************

	for (int z = 0; (current_position - params.baseline) < final_pos; z++)
	{
		printProgBar((int) ((z / number_of_iterations) * 100));
		cout << z << " of " << (int)(number_of_iterations + 0.5);


		// Update position of pin hole and lasers
		setLasersAndPinHole(&pin_hole, &laser_origin_1, &laser_origin_2, current_position, params);
		current_position += increment;

		
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();
		
		//************* Look for intersection with mesh (PCL + OpenCL) *******************
		// For laser 1
		findPointsMeshLaserIntersectionOpenCL(&openCLData, all_triangles, big_triangles_vec, output_points, output_hits, mesh, laser_origin_1, params, cloud_intersection, 
			&plane_1, aperture_coefficient, inclination_coefficient, max_point_triangle, LASER_1, bounds);
		// For laser 2
		findPointsMeshLaserIntersectionOpenCL(&openCLData, all_triangles, big_triangles_vec, output_points, output_hits, mesh, laser_origin_2, params, cloud_intersection,
			&plane_2, aperture_coefficient, inclination_coefficient, max_point_triangle, LASER_2, bounds);

		//duration<double> timer2 = high_resolution_clock::now() - start;
		//cout << "Total time Intersection:" << timer2.count() * 1000 << endl;


		//************** Take snapshot  **************************************************
		cameraSnapshot(camera, pin_hole, laser_origin_1, laser_origin_2, cloud_intersection, &image, params, size_array, &openCLData, all_triangles, output_points,
			output_hits, max_point_triangle);
	
		// Save snapshot (only for debug) 
		if(snapshot_save_flag)
			imwrite("../imgOut/out_" + to_string(z) + ".png", image);


		//************** Convert image to point cloud ************************************
		imageToCloud(camera, params, plane_1, plane_2, pin_hole, &image, 0, camera.image_height / 2, camera.image_height / 2, cloud_out);


		//************** Make a backup of point cloud that contains (all) intersections **
		for (int i = 0; i < cloud_intersection->size(); i++)
			cloud_intersection_backup->push_back(cloud_intersection->at(i));

		// Clear current intersections
		cloud_intersection->clear();
	}


	// Points of the cloud 
	cout << endl << "Numero di punti cloud rossa opencl: " << cloud_intersection_backup->points.size() << endl;
	cout << endl << "Numero di punti della cloud: " << cloud_out->points.size() << endl;


	// Save clouds 
	if (cloud_intersection_backup->size() > 0)
	{
		if (io::savePCDFileASCII("all_intersection_cloud.pcd", *cloud_intersection_backup))
			PCL_ERROR("Failed to save PCD file\n");
	}
	else
		cerr << "WARNING! Point Cloud intersection is empty" << endl;


	if (cloud_out->size() > 0)
	{
		if (io::savePCDFileASCII("final_cloud.pcd", *cloud_out))
			PCL_ERROR("Failed to save PCD file\n");
	}
	else
		cerr << "WARNING! Point Cloud Final is empty" << endl;
	

	//cloud_out->clear();
	//io::loadPCDFile("final_cloud.pcd", *cloud_out);

	//****************** Visualize cloud *************************************************
	visualization::PCLVisualizer viewer("PCL viewer");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud_intersection_backup);
	viewer.addCoordinateSystem(100, "PCL viewer");
	viewer.addPointCloud<PointXYZRGB>(cloud_intersection_backup, rgb, "Intersection Cloud");
	viewer.addPointCloud<PointXYZ>(cloud_out, "Final Cloud");
	//viewer.addPolygonMesh(mesh, "mesh");
	
	// Print total time of computation 
	cout << endl << "Durata: " << returnTime(high_resolution_clock::now() - start) << endl;

	viewer.spin();


	return 0;
}