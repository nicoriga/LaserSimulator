#pragma once

#define __CL_ENABLE_EXCEPTIONS
#define _CRT_SECURE_NO_DEPRECATE
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


#include <math.h>
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
#include <CL/cl.hpp>

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

using namespace cv;
using namespace std;
using namespace pcl;
using boost::chrono::high_resolution_clock;
using boost::chrono::duration;

Eigen::Matrix<double, 3, 1> typedef Vector3d;

/*************************** STRUCTS for the program *************************/
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
	int roi_1_start;
	int roi_2_start;
	int roi_dimension;
	bool distortion_flag;
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
	//cl::Buffer device_triangle_array;
	cl::Buffer device_array_laser;
	cl::Buffer device_output_points;
	cl::Buffer device_output_hits;

	//size_t triangles_size;
	size_t array_laser_size;
	size_t points_size;
	size_t hits_size;

	cl::Context context;
	cl::CommandQueue queue;
	cl::Kernel kernel;
	cl::Program program_;

	vector<cl::Device> devices;
	vector<cl::Platform> platforms;
};


/************************* FUNCTIONS for the program **************************/

void readParamsFromXML(Camera *camera, SimulationParams *params, bool *snapshot_save_flag, string *path_read_file, string *path_save_file);

void updateMinMax(PointXYZRGB point, MeshBounds *bounds);

void calculateBoundaries(const SimulationParams &params, PolygonMesh mesh, MeshBounds *bounds);

void setInitialPosition(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, const SimulationParams &params, const MeshBounds &bounds);

void setLasersAndPinHole(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, float current_position, const SimulationParams &params);

void getPlaneCoefficents(const PointXYZ &laser, Plane *plane, int laser_number, const SimulationParams &params);

int fillSliceWithTriangles(PolygonMesh mesh, vector<int> *triangles_index, const Plane &origin_plane, int laser_number, float slice_length, const int slice_number, const SimulationParams &params);

void createTrianglesArray(const PolygonMesh &mesh, Triangle* triangles, vector<int> *triangles_index, int num_triangles_index_array);

void createSliceBoundArray(int *slice_bound, vector<int> *triangles_index, int * total_triangle, int slice_number);

int getSliceIndex(const PointXYZ &laser_point, const Plane &origin_plane, int laser_number, float slice_length, const int slice_number, const SimulationParams &params);

void initializeOpenCL(OpenCLDATA* data, Triangle* array_laser, int array_lenght, int array_size_hits);

void computeOpenCL(OpenCLDATA* data, Vec3* output_points, uchar* output_hits, int start_index, int array_lenght, const Vec3 &ray_origin, const Vec3 &ray_direction);

void getIntersectionOpenCL(OpenCLDATA* data, Vec3* output_points, uchar* output_hits, const PolygonMesh &mesh, const PointXYZ &laser_point, const SimulationParams &params, PointCloud<PointXYZRGB>::Ptr cloud_intersection, const Plane &origin_plane, Plane* plane,
	const int laser_number, const MeshBounds &bounds, float slice_length, int slice_number, const int *slice_bound);

bool isOccluded(const PointXYZRGB &point, const PointXYZ &pin_hole, float* max_point_triangle, int polygon_size, OpenCLDATA* openCLData, Triangle* all_triangles,
	Vec3* output_points, uchar* output_hits);

void cameraSnapshot(const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_1, const PointXYZ &laser_2, PointCloud<PointXYZRGB>::Ptr cloud_intersection,
	Mat* img, const SimulationParams &params, int polygon_size, OpenCLDATA* openCLData,  Vec3* output_points,
	uchar* output_hits, float* max_point_triangle);

void imageToCloud(Camera &camera, const SimulationParams &params, const Plane &plane_1, const Plane &plane_2, const PointXYZ &pin_hole, Mat* image, PointCloud<PointXYZ>::Ptr cloud_out);

void loadMesh(string path_file, PolygonMesh *mesh);

void saveCloud(string cloud_name, PointCloud<PointXYZ>::Ptr cloud);

string printProgBar(int percent);

string returnTime(duration<double> timer);

void getScanCycleParams(const SimulationParams &params, const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_origin_1, const PointXYZ &laser_origin_2,
	const MeshBounds &bounds, float *increment, float *current_position, float *number_of_iterations, float *final_pos);

string getMeshBoundsValues(const MeshBounds &bounds);