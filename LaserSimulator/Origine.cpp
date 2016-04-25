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
#include <pcl/common/angles.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <CL\cl2.hpp>
#include <thread>

using namespace cv;
using namespace std;
using namespace pcl;
using boost::chrono::high_resolution_clock;
using boost::chrono::duration;


// OpenCL parameter
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

struct SimulationParams
{
	float baseline;
	float height_to_mesh;
	float laser_aperture;
	float laser_inclination;
	float scan_speed;
	float inclination_coefficient;
	float aperture_coefficient;
	int number_of_line;
	int scan_direction;
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
	cl::Buffer device_big_triangle_array;
	cl::Buffer device_output_points;
	cl::Buffer device_output_hits;

	size_t triangles_size;
	size_t big_triangles_size;
	size_t points_size;
	size_t hits_size;

	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
	cl::Program program_;

	vector<cl::Device> devices;
	vector<cl::Platform> platforms;
};



bool isBigTriangle(const Triangle &triangle, float projection_distance) {

	float diff_x, diff_y, diff_z;
	Vec3 ret;

	diff_x = triangle.vertex_1.points[0] - triangle.vertex_2.points[0];
	diff_y = triangle.vertex_1.points[1] - triangle.vertex_2.points[1];
	diff_z = triangle.vertex_1.points[2] - triangle.vertex_2.points[2];

	ret.points[0] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	diff_x = triangle.vertex_1.points[0] - triangle.vertex_3.points[0];
	diff_y = triangle.vertex_1.points[1] - triangle.vertex_3.points[1];
	diff_z = triangle.vertex_1.points[2] - triangle.vertex_3.points[2];

	ret.points[1] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	diff_x = triangle.vertex_2.points[0] - triangle.vertex_3.points[0];
	diff_y = triangle.vertex_2.points[1] - triangle.vertex_3.points[1];
	diff_z = triangle.vertex_2.points[2] - triangle.vertex_3.points[2];

	ret.points[2] = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

	return (ret.points[0] > projection_distance || ret.points[1] > projection_distance || ret.points[2] > projection_distance);
}

void readParamsFromXML(Camera *camera, SimulationParams *params, bool *snapshot_save_flag, string *path_read_file, string *path_save_file)
{
	// Read input parameters from xml file
	FileStorage fs("laser_simulator_params.xml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["path_read_file"] >> *path_read_file;
		fs["path_save_file"] >> *path_save_file;
		fs["baseline"] >> params->baseline;
		fs["height_to_mesh"] >> params->height_to_mesh;
		fs["laser_aperture"] >> params->laser_aperture;
		fs["laser_inclination"] >> params->laser_inclination;
		fs["number_of_line"] >> params->number_of_line;
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
		exit(1);
	}


	if (params->scan_direction != DIRECTION_SCAN_AXIS_X && params->scan_direction != DIRECTION_SCAN_AXIS_Y)
	{
		params->scan_direction = DIRECTION_SCAN_AXIS_Y;
		cout << "WARNING: Direzione di scansione non valida (verra' impostato automaticamente l'asse Y)" << endl << endl;
	}

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

	// Calculate inclination and angular coefficients
	params->inclination_coefficient = tan(deg2rad(90.f - params->laser_inclination));
	params->aperture_coefficient = tan(deg2rad(params->laser_aperture / 2.f));
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

	// Convert mesh in a point cloud (only vertex)
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// Search minimum and maximum points on X, Y and Z axis
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

void setInitialPosition(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, const SimulationParams &params, const MeshBounds &bounds) {
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		laser_origin_1->z = bounds.max_z + params.height_to_mesh;
		laser_origin_1->x = (bounds.max_x + bounds.min_x) / 2;
		laser_origin_1->y = bounds.min_y - (laser_origin_1->z - bounds.min_z) * params.inclination_coefficient;

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
		laser_origin_1->x = bounds.min_x - (laser_origin_1->z - bounds.min_z) * params.inclination_coefficient;

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

float getLinePlaneIntersection(const PointXYZ &source, const Vector3d &direction, float plane_coordinate, int scan_direction) {
	if (scan_direction == DIRECTION_SCAN_AXIS_Y)
		return direction[1] * (plane_coordinate - source.z) / direction[2] + source.y;

	if (scan_direction == DIRECTION_SCAN_AXIS_X)
		return direction[0] * (plane_coordinate - source.z) / direction[2] + source.x;

	return 0;
}

void getPlaneCoefficents(const PointXYZ &laser, const Vector3d &line_1, const Vector3d &line_2, Plane *p) {
	Vector3d plane_normal = line_1.cross(line_2);
	p->A = plane_normal[0];
	p->B = plane_normal[1];
	p->C = plane_normal[2];
	p->D = -plane_normal[0] * laser.x - plane_normal[1] * laser.y - plane_normal[2] * laser.z;
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

void findBigTriangles(const PolygonMesh &mesh, const MeshBounds &bounds, const SimulationParams &params, vector<Triangle> *big_triangles_vec,
	vector<int> *big_triangles_index, int size_array)
{
	PointCloud<PointXYZ> mesh_vertices;
	PointXYZ point;
	Triangle triangle;

	float projection_distance = (bounds.max_z - bounds.min_z) * params.inclination_coefficient;

	fromPCLPointCloud2(mesh.cloud, mesh_vertices);

	for (int i = 0; i < size_array; i++)
	{
		// Take Vertex 1
		point = mesh_vertices.points[mesh.polygons[i].vertices[0]];
		triangle.vertex_1.points[X] = point.x;
		triangle.vertex_1.points[Y] = point.y;
		triangle.vertex_1.points[Z] = point.z;

		// Take Vertex 2
		point = mesh_vertices.points[mesh.polygons[i].vertices[1]];
		triangle.vertex_2.points[X] = point.x;
		triangle.vertex_2.points[Y] = point.y;
		triangle.vertex_2.points[Z] = point.z;

		// Take Vertex 3
		point = mesh_vertices.points[mesh.polygons[i].vertices[2]];
		triangle.vertex_3.points[X] = point.x;
		triangle.vertex_3.points[Y] = point.y;
		triangle.vertex_3.points[Z] = point.z;

		// Check if the triangle is a "Big Triangle" and save its index
		if (isBigTriangle(triangle, projection_distance))
		{
			big_triangles_vec->push_back(triangle);
			big_triangles_index->push_back(i);
		}
	}
}

void removeDuplicate(float* max_point_triangle, int* max_point_triangle_index, int max_point_array_dimension, vector<int> &big_triangles_index)
{
	int current_position = 0;

	for (int i = 0; i < big_triangles_index.size(); i++)
	{
		for (int j = 0; j < max_point_array_dimension; j++)
		{
			// Find the duplicate
			if (big_triangles_index[i] != max_point_triangle_index[j])
			{
				max_point_triangle[current_position] = max_point_triangle[j];
				max_point_triangle_index[current_position] = max_point_triangle_index[j];
				current_position++;
				break;
			}
		}
	}
}

void createAllTriangleArray(const PolygonMesh &mesh, Triangle* triangles, int* max_point_triangle_index, int size_array)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	PointXYZ tmp;

	for (int k = 0; k < size_array; k++)
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

void initializeOpenCL(OpenCLDATA* data, Triangle* triangle_array, int array_lenght, Triangle* big_triangle_array, int big_array_lenght, int array_size_hits) {
	cl_int err = CL_SUCCESS;

	try {
		// Query platforms
		cl::Platform::get(&data->platforms);
		if (data->platforms.size() == 0)
		{
			cerr << "OpenCL error: Platform size 0" << endl;
			exit(1);
		}

		// Get list of devices on default platform and create context
		cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(data->platforms[0])(), 0 };
		//data->context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
		data->context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
		data->devices = data->context.getInfo<CL_CONTEXT_DEVICES>();

		// Create command queue for first device
		data->queue = cl::CommandQueue(data->context, data->devices[0], 0, &err);

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
		data->program_ = cl::Program(data->context, kernelSource);
		err = data->program_.build(data->devices);

		free(kernelSource);

		// Size, in bytes, of each vector
		data->triangles_size = array_lenght * sizeof(Triangle);
		data->big_triangles_size = big_array_lenght * sizeof(Triangle);
		data->points_size = array_size_hits * sizeof(Vec3);
		data->hits_size = array_size_hits * sizeof(uchar);

		// Create device memory buffers
		data->device_triangle_array = cl::Buffer(data->context, CL_MEM_READ_ONLY, data->triangles_size);
		data->device_big_triangle_array = cl::Buffer(data->context, CL_MEM_READ_ONLY, data->big_triangles_size);
		data->device_output_points = cl::Buffer(data->context, CL_MEM_WRITE_ONLY, data->points_size);
		data->device_output_hits = cl::Buffer(data->context, CL_MEM_WRITE_ONLY, data->hits_size);

		// Bind memory buffers
		data->queue.enqueueWriteBuffer(data->device_triangle_array, CL_TRUE, 0, data->triangles_size, triangle_array);
		data->queue.enqueueWriteBuffer(data->device_big_triangle_array, CL_TRUE, 0, data->big_triangles_size, big_triangle_array);

		// Create kernel object
		data->kernel = cl::Kernel(data->program_, "kernelTriangleIntersection", &err);

		// Bind kernel arguments to kernel
		data->kernel.setArg(1, data->device_output_points);
		data->kernel.setArg(2, data->device_output_hits);

	}
	catch (...) {
		cerr << "OpenCL error" << endl;
		exit(1);
	}

}

void computeOpenCL(OpenCLDATA* data, Vec3* output_points, uchar* output_hits, int start_index, int array_lenght, const Vec3 &ray_origin, const Vec3 &ray_direction, bool big) {

	//high_resolution_clock::time_point start;
	//start = high_resolution_clock::now();

	if (big)
		data->kernel.setArg(0, data->device_big_triangle_array);

	else
		data->kernel.setArg(0, data->device_triangle_array);

	data->kernel.setArg(3, start_index);
	data->kernel.setArg(4, array_lenght);
	data->kernel.setArg(5, ray_origin);
	data->kernel.setArg(6, ray_direction);

	// Number of work items in each local work group

	cl::NDRange localSize(LOCAL_SIZE, 1, 1);
	// Number of total work items - localSize must be divisor
	int global_size = (int)(ceil((array_lenght / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
	cl::NDRange globalSize(global_size, 1, 1);

	// Enqueue kernel
	cl::Event event;
	data->queue.enqueueNDRangeKernel(data->kernel, cl::NullRange, globalSize, localSize, NULL, &event);

	// Block until kernel completion
	event.wait();

	// Read back device_output_point, device_output_hit
	data->queue.enqueueReadBuffer(data->device_output_points, CL_TRUE, 0, data->points_size, output_points);
	data->queue.enqueueReadBuffer(data->device_output_hits, CL_TRUE, 0, data->hits_size, output_hits);

	//duration<double> timer = high_resolution_clock::now() - start;
	//cout << "Buffer output copied OpenCL:" << timer.count() * 1000 << endl;
}

void getIntersectionOpenCL(OpenCLDATA* data, Triangle* all_triangles, Vec3* output_points, uchar* output_hits,
	const PolygonMesh &mesh, const PointXYZ &laser_point, const SimulationParams &params, PointCloud<PointXYZRGB>::Ptr cloud_intersection, Plane* plane,
	float* max_point_triangle, const int laser_number, const MeshBounds &bounds, int size_array, int size_big_array)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	int array_size_hits = (int)(ceil(size_array / (float)RUN));

	float ray_density = (params.aperture_coefficient * 2) / params.number_of_line;

	int d1, d2;
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		d1 = 0;
		d2 = 1;
		Vector3d line_1(-params.aperture_coefficient + 0 * ray_density, laser_number * params.inclination_coefficient, -1);
		Vector3d line_2(-params.aperture_coefficient + 10 * ray_density, laser_number * params.inclination_coefficient, -1);
		getPlaneCoefficents(laser_point, line_1, line_2, plane);
	}

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		d1 = 1;
		d2 = 0;
		Vector3d line_1(laser_number * params.inclination_coefficient, -params.aperture_coefficient + 0 * ray_density, -1);
		Vector3d line_2(laser_number * params.inclination_coefficient, -params.aperture_coefficient + 10 * ray_density, -1);
		getPlaneCoefficents(laser_point, line_1, line_2, plane);
	}

	Vector3d laser_direction;
	laser_direction[d1] = -params.aperture_coefficient;
	laser_direction[d2] = laser_number * params.inclination_coefficient;
	laser_direction[2] = -1;
	float laser_intersect_min_z = getLinePlaneIntersection(laser_point, laser_direction, bounds.min_z, params.scan_direction);
	float laser_intersect_max_z = getLinePlaneIntersection(laser_point, laser_direction, bounds.max_z, params.scan_direction);

	int lower_bound, upper_bound;

	// Bounds calculated due to laser
	switch (laser_number)
	{
	case (LASER_1):
		lower_bound = getLowerBound(max_point_triangle, size_array, laser_intersect_max_z);
		upper_bound = getUpperBound(max_point_triangle, size_array, laser_intersect_min_z);
		break;

	case (LASER_2):
		lower_bound = getLowerBound(max_point_triangle, size_array, laser_intersect_min_z);
		upper_bound = getUpperBound(max_point_triangle, size_array, laser_intersect_max_z);
		break;
	}


	int diff = upper_bound - lower_bound;

	for (int j = 0; j < params.number_of_line; j++)
	{
		//high_resolution_clock::time_point start;
		//start = high_resolution_clock::now();

		PointXYZ tmp;
		Vertices triangle;
		Vector3d vertex_1, vertex_2, vertex_3;
		Vector3d intersection_point;
		Vec3 ray_origin, ray_direction;
		PointXYZRGB first_intersec;

		ray_origin.points[X] = laser_point.x;
		ray_origin.points[Y] = laser_point.y;
		ray_origin.points[Z] = laser_point.z;

		float i = -params.aperture_coefficient + j * ray_density;

		first_intersec.z = VTK_FLOAT_MIN;

		ray_direction.points[d1] = i;
		ray_direction.points[d2] = laser_number * params.inclination_coefficient;
		ray_direction.points[Z] = -1;

		if (diff > 0)
		{
			computeOpenCL(data, output_points, output_hits, lower_bound, diff, ray_origin, ray_direction, FALSE);

			int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
			for (int h = 0; h < n_max; h++)
			{
				if (output_hits[h] == 1)
				{
					//++hit_number;
					if (output_points[h].points[Z] >= first_intersec.z)
					{
						first_intersec.x = output_points[h].points[X];
						first_intersec.y = output_points[h].points[Y];
						first_intersec.z = output_points[h].points[Z];
						first_intersec.r = 255;
						first_intersec.g = 0;
						first_intersec.b = 0;
					}
				}
			}
		}

		computeOpenCL(data, output_points, output_hits, 0, size_big_array, ray_origin, ray_direction, TRUE);

		int n = (int)(ceil((size_big_array / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
		for (int h = 0; h < n; h++)
		{
			if (output_hits[h] == 1)
			{
				//++hit_number;

				if (output_points[h].points[Z] >= first_intersec.z)
				{
					first_intersec.x = output_points[h].points[X];
					first_intersec.y = output_points[h].points[Y];
					first_intersec.z = output_points[h].points[Z];
					first_intersec.r = 255;
					first_intersec.g = 0;
					first_intersec.b = 0;
				}

			}
		}

		if (first_intersec.z > VTK_FLOAT_MIN)
			cloud_intersection->push_back(first_intersec);

		//duration<double> timer3 = high_resolution_clock::now() - start;
		//cout << "Total time cycle ray intersection OpenCL:" << timer3.count() * 1000 << endl;
	}
}

bool isOccluded(const PointXYZRGB &point, const PointXYZ &pin_hole, float* max_point_triangle, int polygon_size, OpenCLDATA* openCLData, Triangle* all_triangles,
	Vec3* output_points, uchar* output_hits) {
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
		computeOpenCL(openCLData, output_points, output_hits, lower_bound, diff, origin, direction, FALSE);

		int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
		for (int k = 0; k < n_max; k++)
		{
			if (output_hits[k] == 1)
				return TRUE;
		}
	}

	return FALSE;
}

void cameraSnapshot(const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_1, const PointXYZ &laser_2, PointCloud<PointXYZRGB>::Ptr cloud_intersection,
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

	// Camera
	c.x = 0;
	c.y = 0;
	c.z = 0;
	cloud_target->push_back(c);

	// Laser 1
	p1.x = 0;
	p1.y = -params.baseline;
	p1.z = 0;

	// Laser 2
	p2.x = 0;
	p2.y = params.baseline;
	p2.z = 0;


	cloud_target->push_back(p1);
	cloud_target->push_back(p2);

	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>  transEst;
	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>::Matrix4 trans;
	transEst.estimateRigidTransformation(*cloud_src, *cloud_target, trans);

	vector<Point3d> points;
	vector<Point2d> output_point;

	for (int i = 0; i < cloud_intersection->size(); i++) {
		Eigen::Vector4f v_point, v_point_final;
		v_point[0] = cloud_intersection->points[i].x;
		v_point[1] = cloud_intersection->points[i].y;
		v_point[2] = cloud_intersection->points[i].z;
		v_point[3] = 1;
		v_point_final = trans * v_point;

		v_point_final[2] = -v_point_final[2];

		Point3f p(v_point_final[0], v_point_final[1], v_point_final[2]);

		points.push_back(p);
	}

	if (cloud_intersection->size() > 0) {
		projectPoints(points, Mat::zeros(3, 1, CV_64F), Mat::zeros(3, 1, CV_64F), camera.camera_matrix, camera.distortion, output_point);
		Point2d pixel;
		for (int i = 0; i < output_point.size(); i++) {
			pixel = output_point.at(i);
			pixel.x += 0.5;
			pixel.y += 0.5;

			if ((pixel.y >= 0) && (pixel.y < image.rows) && (pixel.x >= 0) && (pixel.x < image.cols))
			{
				if (!(isOccluded(cloud_intersection->at(i), pin_hole, max_point_triangle, polygon_size, openCLData, all_triangles, output_points, output_hits)))
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

void loadMesh(string path_file, PolygonMesh *mesh)
{
	// Load STL file as a PolygonMesh
	if (io::loadPolygonFileSTL(path_file, *mesh) == 0)
	{
		PCL_ERROR("Failed to load STL file\n");
		exit(1);
	}
}

void saveCloud(string cloud_name, PointCloud<PointXYZ>::Ptr cloud)
{
	if (cloud->size() > 0)
	{
		try
		{
			io::savePCDFileASCII(cloud_name, *cloud);
		}
		catch (pcl::IOException)
		{
			PCL_ERROR("Failed to save PCD file\n");
		}

	}
	else
		cerr << "WARNING! Point Cloud Final is empty" << endl;
}

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
	int sec = (int)timer.count() % 60;
	string seconds = sec < 10 ? "0" + to_string(sec) : to_string(sec);
	string minutes = to_string((int)(((int)timer.count() / 60) % 60));

	return minutes + ":" + seconds + " s";
}

void getScanCycleParams(const SimulationParams &params, const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_origin_1, const PointXYZ &laser_origin_2,
	const MeshBounds &bounds, float *increment, float *current_position, float *number_of_iterations, float *final_pos)
{
	// Step size (in mm) between two snapshots
	*increment = params.scan_speed / camera.fps;

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		*current_position = pin_hole.x;
		*final_pos = bounds.max_x - (bounds.min_x - laser_origin_2.x);
		*number_of_iterations = (*final_pos - laser_origin_1.x) / *increment;
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		*current_position = pin_hole.y;
		*final_pos = bounds.max_y + (bounds.min_y - laser_origin_2.y);
		*number_of_iterations = (*final_pos - laser_origin_1.y) / *increment;
	}
}

string getMeshBoundsValues(const MeshBounds &bounds)
{
	stringstream stream;
	stream << "X: [" << bounds.min_x << ", " << bounds.max_x << "]        ";
	stream << "Y: [" << bounds.min_y << ", " << bounds.max_y << "]        ";
	stream << "Z: [" << bounds.min_z << ", " << bounds.max_z << "]" << endl << endl;
	
	return stream.str();
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

int main(int argc, char** argv)
{
	PolygonMesh mesh;
	MeshBounds bounds;
	Camera camera;
	SimulationParams params;
	bool snapshot_save_flag;
	string path_read_file, path_save_file;
	OpenCLDATA data;

	// Origin point of laser 1, laser 2 and camera pin hole
	PointXYZ laser_origin_1, laser_origin_2, pin_hole;


	//********* Read data from XML parameters file ***************************************
	readParamsFromXML(&camera, &params, &snapshot_save_flag, &path_read_file, &path_save_file);

	// Starting time counter
	high_resolution_clock::time_point start = high_resolution_clock::now();


	//********* Load mesh *****************************************************************
	loadMesh(path_read_file, &mesh);
	cout << "Dimensione della mesh (n. triangoli): " << mesh.polygons.size() << endl << endl;

	// Arrays used to optimize the computation of intersections
	float *max_point_triangle = new float[mesh.polygons.size()];
	int *max_point_triangle_index = new int[mesh.polygons.size()];


	//********** Find minimum and mixiumum points of 3 axis and fill *********************
	//********** arrays used to find maximum value on the direction axis *****************
	calculateBoundariesAndArrayMax(params, mesh, max_point_triangle_index, max_point_triangle, &bounds);


	//********** Print minimum and maximum points of mesh ********************************
	cout << "Estremi della mesh:" << endl << getMeshBoundsValues(bounds);
	/*cout << "X: [" << bounds.min_x << ", " << bounds.max_x << "]        ";
	cout << "Y: [" << bounds.min_y << ", " << bounds.max_y << "]        ";
	cout << "Z: [" << bounds.min_z << ", " << bounds.max_z << "]" << endl << endl;*/

	//************************ Find "big" triangles **************************************
	vector<Triangle> big_triangles_vec;
	vector<int> big_triangles_index;
	
	findBigTriangles(mesh, bounds,params, &big_triangles_vec, &big_triangles_index, mesh.polygons.size());

	// Put big triangles in a Triangle array
	int big_array_size = big_triangles_vec.size();
	Triangle *big_triangles = new Triangle[big_array_size];
	for (int i = 0; i < big_array_size; i++)
		big_triangles[i] = big_triangles_vec[i];
	
	cout << "Numero triangoli \"grandi\": " << big_array_size << endl << endl;

	// Remove "big" triangles from all triangles array
	int array_size = mesh.polygons.size() - big_triangles_index.size();
	removeDuplicate(max_point_triangle, max_point_triangle_index, mesh.polygons.size(), big_triangles_index);

	// Sort arrays to have more efficency in the search
	float *tmp_a = new float[mesh.polygons.size()];
	int *tmp_b = new int[mesh.polygons.size()];
	arraysMergesort(max_point_triangle, max_point_triangle_index, 0, array_size - 1, tmp_a, tmp_b);
	delete[] tmp_a, tmp_b;

	//**************** Initialize OpenCL *************************************************
	Triangle *all_triangles = new Triangle[array_size];
	createAllTriangleArray(mesh, all_triangles, max_point_triangle_index, array_size);

	int array_size_hits = (int) (ceil(array_size / (float)RUN));
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	initializeOpenCL(&data, all_triangles, array_size, big_triangles, big_array_size, array_size_hits);


	//**************** Set initial position for camera and lasers *****************
	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	// Set current position, calculate final position and number of iterations
	float increment, current_position, number_of_iterations, final_pos;
	getScanCycleParams(params, camera, pin_hole, laser_origin_1, laser_origin_2, bounds, &increment, &current_position, &number_of_iterations, &final_pos);

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection_backup(new PointCloud<PointXYZRGB>);
	Plane plane_1, plane_2;
	Mat image;

	/* Disegno i laser
	if (scan_direction == DIRECTION_SCAN_AXIS_X)
	{
	drawLine(cloud_intersection_backup, laser_origin_1, Eigen::Vector3f(-inclination_coefficient, -0, -1), 2000);
	drawLine(cloud_intersection_backup, laser_origin_2, Eigen::Vector3f(inclination_coefficient, 0, -1), 2000);
	}
	if (scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
	drawLine(cloud_intersection_backup, laser_origin_1, Eigen::Vector3f(0, -inclination_coefficient, -1), 2000);
	drawLine(cloud_intersection_backup, laser_origin_2, Eigen::Vector3f(0, inclination_coefficient, -1), 2000);
	}*/


	//************CORE OF THE PROJECT: this cycle simulates the laser scan. *****************
	//*********** In every iteration finds intersection with mesh, take a camera snapshot ***
	//*********** and reconstruct the points in the 3D space ********************************

	for (int z = 0; (current_position - params.baseline) < final_pos; z++)
	{
		// Print progression bar and number of iteration completed
		printProgBar((int) ((z / number_of_iterations) * 100 + 0.5));
		cout << z << " di " << (int) (number_of_iterations);


		// Update position of pin hole and lasers
		setLasersAndPinHole(&pin_hole, &laser_origin_1, &laser_origin_2, current_position, params);
		current_position += increment;


		//************* Look for intersection with mesh *******************
		// For laser 1
		getIntersectionOpenCL(&data, all_triangles, output_points, output_hits, mesh, laser_origin_1, params, cloud_intersection,
			&plane_1, max_point_triangle, LASER_1, bounds, array_size, big_array_size);
		// For laser 2
		getIntersectionOpenCL(&data, all_triangles, output_points, output_hits, mesh, laser_origin_2, params, cloud_intersection,
			&plane_2, max_point_triangle, LASER_2, bounds, array_size, big_array_size);


		//************** Take snapshot  **************************************************
		cameraSnapshot(camera, pin_hole, laser_origin_1, laser_origin_2, cloud_intersection, &image, params, array_size, &data, all_triangles, output_points,
			output_hits, max_point_triangle);

		// Save snapshot (only for debug) 
		if (snapshot_save_flag)
			imwrite("../imgOut/out_" + to_string(z) + ".png", image);

		//************** Convert image to point cloud ************************************
		imageToCloud(camera, params, plane_1, plane_2, pin_hole, &image, 0, camera.image_height / 2, camera.image_height / 2, cloud_out);


		// Make a backup of point cloud that contains (all) intersections
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
		if (io::savePCDFileASCII("../result/all_intersection_cloud.pcd", *cloud_intersection_backup))
			PCL_ERROR("Failed to save PCD file\n");
	}
	else
		cerr << "WARNING! Point Cloud intersection is empty" << endl;

	saveCloud(path_save_file, cloud_out);


	//****************** Visualize cloud *************************************************
	visualization::PCLVisualizer viewer("PCL viewer");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud_intersection_backup);
	viewer.addCoordinateSystem(100, "PCL viewer");
	viewer.addPointCloud<PointXYZRGB>(cloud_intersection_backup, rgb, "Intersection Cloud");
	viewer.addPointCloud<PointXYZ>(cloud_out, "Cloud");

	// Print total time of computation 
	cout << endl << "Durata: " << returnTime(high_resolution_clock::now() - start) << endl;

	viewer.spin();


	return 0;
}