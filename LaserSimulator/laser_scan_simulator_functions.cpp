/*
* LaserSimulator
* Created on: 02/02/2016
* Last Update: 21/04/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/

#include "laser_scan_simulator_functions.h"



bool isBigTriangle(const Triangle &triangle, float projection_distance) 
{
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
	camera->distortion = Mat::zeros(5, 1, CV_64F);

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
		fs["man_thresh"] >> params->man_thresh;
		fs["number_of_line"] >> params->number_of_line;
		fs["scan_speed"] >> params->scan_speed;
		fs["scan_direction"] >> params->scan_direction;
		fs["distortion_flag"] >> params->distortion_flag;
		fs["snapshot_save_flag"] >> *snapshot_save_flag;
		fs["roi_1_start"] >> params->roi_1_start;
		fs["roi_2_start"] >> params->roi_2_start;
		fs["roi_dimension"] >> params->roi_dimension;
		fs["camera_fps"] >> camera->fps;
		fs["image_width"] >> camera->image_width;
		fs["image_height"] >> camera->image_height;
		fs["pixel_dimension"] >> camera->pixel_dimension;
		fs["camera_matrix"] >> camera->camera_matrix;

		if(params->distortion_flag)
			fs["camera_distortion"] >> camera->distortion;
	}
	else
	{
		cerr << "Errore: Lettura file XML fallita" << endl;
		exit(1);
	}


	if (params->scan_direction != DIRECTION_SCAN_AXIS_X && params->scan_direction != DIRECTION_SCAN_AXIS_Y)
	{
		params->scan_direction = DIRECTION_SCAN_AXIS_Y;
		cerr << "WARNING: Direzione di scansione non valida (verra' impostato automaticamente l'asse Y)" << endl << endl;
	}

	if ((params->roi_1_start + params->roi_dimension) > params->roi_2_start || (params->roi_2_start + params->roi_dimension) > camera->image_height)
	{
		cerr << "Errore: Valori ROI non validi" << endl;
		exit(1);
	}

	if (params->scan_speed < 100)
	{
		params->scan_speed = 100.f;
		cerr << "WARNING: Velocita' di scansione inferiore a 100 (verra' impostata automaticamente a 100)" << endl << endl;
	}

	if (params->scan_speed > 1000)
	{
		params->scan_speed = 1000.f;
		cerr << "WARNING: Velocita' di scansione superiore a 1000 (verra' impostata automaticamente a 1000)" << endl << endl;
	}

	if (params->baseline < 500)
	{
		params->baseline = 500.f;
		cerr << "WARNING: Baseline inferiore a 500 (verra' impostata automaticamente a 500)" << endl << endl;
	}

	if (params->baseline > 800)
	{
		params->baseline = 800.f;
		cerr << "WARNING: Baseline superiore a 800 (verra' impostata automaticamente a 800)" << endl << endl;
	}

	if (camera->fps < 100)
	{
		camera->fps = 100.f;
		cerr << "WARNING: FPS della camera inferiori a 100 (verranno impostati automaticamente a 100)" << endl << endl;
	}

	if (camera->fps > 500)
	{
		camera->fps = 500.f;
		cerr << "WARNING: FPS della camera superiori a 500 (verranno impostati automaticamente a 500)" << endl << endl;
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

void sortArrays(float *a, int* b, int array_size)
{
	float *tmp_a = new float[array_size];
	int *tmp_b = new int[array_size];
	arraysMergesort(a, b, 0, array_size - 1, tmp_a, tmp_b);
	delete[] tmp_a, tmp_b;
}

void updateMinMax(PointXYZRGB point, MeshBounds *bounds) 
{
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

void calculateBoundariesAndArrayMax(const SimulationParams &params, PolygonMesh mesh, int* max_point_triangle_index, float* max_point_triangle, MeshBounds *bounds) 
{
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

void setInitialPosition(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, const SimulationParams &params, const MeshBounds &bounds) 
{
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

void setLasersAndPinHole(PointXYZ* pin_hole, PointXYZ* laser_origin_1, PointXYZ* laser_origin_2, float current_position, const SimulationParams &params) 
{
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

float getLinePlaneIntersection(const PointXYZ &source, const Vector3d &direction, float plane_coordinate, int scan_direction) 
{
	if (scan_direction == DIRECTION_SCAN_AXIS_Y)
		return direction[1] * (plane_coordinate - source.z) / direction[2] + source.y;

	if (scan_direction == DIRECTION_SCAN_AXIS_X)
		return direction[0] * (plane_coordinate - source.z) / direction[2] + source.x;

	return 0;
}

void getPlaneCoefficents(const PointXYZ &laser, const Vector3d &line_1, const Vector3d &line_2, Plane *p) 
{
	Vector3d plane_normal = line_1.cross(line_2);
	p->A = plane_normal[0];
	p->B = plane_normal[1];
	p->C = plane_normal[2];
	p->D = -plane_normal[0] * laser.x - plane_normal[1] * laser.y - plane_normal[2] * laser.z;
}

int getLowerBound(float* array_points, int array_size, float threshold) 
{
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

int getUpperBound(float* array_points, int array_size, float threshold, const SimulationParams &params)
{
	int index = 0;

	for (int i = array_size - 1; i > 0; i--)
	{
		if (array_points[i] < threshold + params.man_thresh) {
			index = i;
			break;
		}
	}

	return index;
}

int getUpperBound(float* array_points, int array_size, float threshold)
{
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
	int diff_size = max_point_array_dimension - big_triangles_index.size();
	float *max_removed = new float[diff_size];
	int *max_index_removed = new int[diff_size];

	for (int i = 0; i < max_point_array_dimension; i++)
	{
		for (int j = 0; j < big_triangles_index.size(); j++)
		{
			if (max_point_triangle_index[i] == big_triangles_index[j])
			{
				max_point_triangle_index[i] = -1;
			}
		}
	}

	int count = 0;
	for (int i = 0; i < max_point_array_dimension; i++)
	{
		if (max_point_triangle_index[i] != -1)
		{
			max_removed[count] = max_point_triangle[i];
			max_index_removed[count] = max_point_triangle_index[i];
			count++;
		}
	}

	for (int i = 0; i < diff_size; i++)
	{
		max_point_triangle[i] = max_removed[i];
		max_point_triangle_index[i] = max_index_removed[i];
	}


	/*int current_position = 0;

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
	}*/
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

void initializeOpenCL(OpenCLDATA* data, Triangle* triangle_array, int array_lenght, Triangle* big_triangle_array, int big_array_lenght, int array_size_hits) 
{
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
		data->context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
		data->devices = data->context.getInfo<CL_CONTEXT_DEVICES>();

		// Create command queue for first device
		data->queue = cl::CommandQueue(data->context, data->devices[0], 0, &err);

		FILE* programHandle;
		size_t kernelSourceSize;
		char *kernelSource;

		// Get size of kernel source
		fopen_s(&programHandle, "IntersectionTriangle.cl", "rb");
		fseek(programHandle, 0, SEEK_END);
		kernelSourceSize = ftell(programHandle);
		rewind(programHandle);

		// Read kernel source into buffer
		kernelSource = (char*)malloc(kernelSourceSize + 1);
		kernelSource[kernelSourceSize] = '\0';
		fread(kernelSource, sizeof(char), kernelSourceSize, programHandle);
		fclose(programHandle);

		// Build kernel from source string
		data->program_ = cl::Program(data->context, kernelSource);
		err = data->program_.build(data->devices);

		if (err != CL_SUCCESS)
		{
			cerr << "OpenCL error" << endl;
			exit(1);
		}
		
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
		data->queue.finish();

		// Create kernel object
		data->kernel = cl::Kernel(data->program_, "kernelTriangleIntersection", &err);

		// Bind kernel arguments to kernel
		err = data->kernel.setArg(1, data->device_output_points);
		err = data->kernel.setArg(2, data->device_output_hits);

		if (err != CL_SUCCESS)
		{
			cerr << "OpenCL error" << endl;
			exit(1);
		}

	}

	catch (cl::Error er)
	{
		cerr << "OpenCL error" << endl;
		exit(1);
	}

}

void computeOpenCL(OpenCLDATA* data, Vec3* output_points, uchar* output_hits, int start_index, int array_lenght, const Vec3 &ray_origin, const Vec3 &ray_direction, bool big) 
{
	cl_int err = CL_SUCCESS;

	if (big)
		err = data->kernel.setArg(0, data->device_big_triangle_array);

	else
		err = data->kernel.setArg(0, data->device_triangle_array);

	err = data->kernel.setArg(3, start_index);
	err = data->kernel.setArg(4, array_lenght);
	err = data->kernel.setArg(5, ray_origin);
	err = data->kernel.setArg(6, ray_direction);

	if (err != CL_SUCCESS)
	{
		cerr << "OpenCL error" << endl;
		exit(1);
	}

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
}

void getIntersectionOpenCL(OpenCLDATA* data, Vec3* output_points, uchar* output_hits,
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
			upper_bound = getUpperBound(max_point_triangle, size_array, laser_intersect_min_z, params);
			break;

		case (LASER_2):
			lower_bound = getLowerBound(max_point_triangle, size_array, laser_intersect_min_z);
			upper_bound = getUpperBound(max_point_triangle, size_array, laser_intersect_max_z, params);
			break;
	}


	int diff = upper_bound - lower_bound;

	for (int j = 0; j < params.number_of_line; j++)
	{
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
	}
}

bool isOccluded(const PointXYZRGB &point, const PointXYZ &pin_hole, float* max_point_triangle, int polygon_size, OpenCLDATA* openCLData, Triangle* all_triangles,
	Vec3* output_points, uchar* output_hits) 
{
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
	uchar* output_hits, float* max_point_triangle) 
{
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

void imageToCloud(Camera &camera, const SimulationParams &params, const Plane &plane_1, const Plane &plane_2, const PointXYZ &pin_hole, Mat* image, PointCloud<PointXYZ>::Ptr cloud_out) 
{
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
	if (params.distortion_flag)
	{
		Mat image_undistort;
		undistort(*image, image_undistort, camera.camera_matrix, camera.distortion);
		flip(image_undistort, *image, 0);
	}
	
	else
		flip(*image, *image, 0);


	// Project the ROI1 points on the first plane
	for (int j = 0; j < image->cols; j++)
	{
		for (int i = params.roi_1_start; i < params.roi_1_start + params.roi_dimension; i++)
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
		for (int i = params.roi_2_start; i < params.roi_2_start + params.roi_dimension; i++)
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
	// Load mesh file as a PolygonMesh
	if (io::loadPolygonFile(path_file, *mesh) == 0)
	{
		PCL_ERROR("Failed to load mesh file\n");
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

string printProgBar(int percent) 
{
	stringstream prog;
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
	prog << "\r" "[" << bar << "] ";
	prog.width(3);
	prog << percent << "%     " << flush;

	return prog.str();
}

string returnTime(duration<double> timer)
{
	int sec = (int) timer.count() % 60;
	string seconds = sec < 10 ? "0" + to_string(sec) : to_string(sec);
	string minutes = to_string((int) timer.count() / 60);

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
		*final_pos = bounds.max_x + (bounds.min_x - laser_origin_2.x);
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
