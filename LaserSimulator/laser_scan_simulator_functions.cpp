/*
* Laser scan simulator
* Created on: 18/02/2016
* Last Update: 09/05/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/

#include "laser_scan_simulator_functions.h"


void readParamsFromXML(Camera *camera, SimulationParams *params, bool *snapshot_save_flag, string *path_read_file, string *path_save_file)
{
	camera->distortion = Mat::zeros(5, 1, CV_64F);

	// Read input parameters from xml file
	FileStorage fs("laser_scan_simulator_params.xml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["path_read_file"] >> *path_read_file;
		fs["path_save_file"] >> *path_save_file;
		fs["baseline"] >> params->baseline;
		fs["height_from_mesh"] >> params->height_from_mesh;
		fs["laser_aperture"] >> params->laser_aperture;
		fs["laser_inclination"] >> params->laser_inclination;
		fs["number_of_line"] >> params->number_of_line;
		fs["scan_speed"] >> params->scan_speed;
		fs["scan_direction"] >> params->scan_direction;
		fs["undistortion_flag"] >> params->undistortion_flag;
		fs["snapshot_save_flag"] >> *snapshot_save_flag;
		fs["roi_1_start"] >> params->roi_1_start;
		fs["roi_2_start"] >> params->roi_2_start;
		fs["roi_dimension"] >> params->roi_dimension;
		fs["camera_fps"] >> camera->fps;
		fs["image_width"] >> camera->image_width;
		fs["image_height"] >> camera->image_height;
		fs["pixel_dimension"] >> camera->pixel_dimension;
		fs["camera_matrix"] >> camera->camera_matrix;

		if(params->undistortion_flag)
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
	params->compl_inclination_coeff = tan(deg2rad(params->laser_inclination));
	params->aperture_coefficient = tan(deg2rad(params->laser_aperture / 2.f));
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

float pointsDistance(PointXYZ point1, PointXYZ point2)
{
	float diff_x = point1.x - point2.x;
	float diff_y = point1.y - point2.y;
	float diff_z = point1.z - point2.z;

	return sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
}

void calculateBoundaries(const PolygonMesh &mesh, MeshBounds *bounds)
{
	PointCloud<PointXYZ> cloud_mesh;
	PointXYZRGB point_1, point_2, point_3;

	// Convert mesh in a point cloud (only vertices)
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// Search minimum and maximum points on X, Y and Z axis
	for (int i = 0; i < mesh.polygons.size(); ++i)
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
	}
}

void setInitialPosition(PointXYZ *pin_hole, PointXYZ *laser_origin_1, PointXYZ *laser_origin_2, const SimulationParams &params, const MeshBounds &bounds) 
{
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		laser_origin_1->z = bounds.max_z + params.height_from_mesh;
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
		laser_origin_1->z = bounds.max_z + params.height_from_mesh;
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

void setLasersAndPinHole(PointXYZ *pin_hole, PointXYZ *laser_origin_1, PointXYZ *laser_origin_2, float current_position, const SimulationParams &params) 
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

void setSliceParams(SliceParams *slice_params, const PointXYZ &laser_origin_1, const PointXYZ &laser_origin_2, const SimulationParams &params, const MeshBounds &bounds)
{
	float total_length = 0;
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
		total_length = bounds.max_y + (bounds.min_y - laser_origin_1.y) - laser_origin_1.y;

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
		total_length = bounds.max_y + (bounds.min_y - laser_origin_1.x) - laser_origin_1.x;

	slice_params->slice_length = total_length / SLICE_NUMBER;
	slice_params->vertical_slice_length = total_length / VERTICAL_SLICE_NUMBER;


	getPlaneCoefficents(laser_origin_1, &slice_params->origin_plane_laser1, LASER_1, params);
	getPlaneCoefficents(laser_origin_2, &slice_params->origin_plane_laser2, LASER_2, params);
	getPlaneCoefficents(laser_origin_1, &slice_params->origin_vertical_plane, VERTICAL_LINE, params);
}

void getPlaneCoefficents(const PointXYZ &laser, Plane *plane, int laser_number, const SimulationParams &params)
{
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		if (laser_number == LASER_1)
		{
			plane->A = params.compl_inclination_coeff;
			plane->B = 0;
			plane->C = 1;
		}
		if (laser_number == LASER_2)
		{
			plane->A = -params.compl_inclination_coeff;
			plane->B = 0;
			plane->C = 1;
		}
		if (laser_number == VERTICAL_LINE)
		{
			plane->A = 1;
			plane->B = 0;
			plane->C = 0;
		}
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		if (laser_number == LASER_1)
		{
			plane->A = 0;
			plane->B = params.compl_inclination_coeff;
			plane->C = 1;
		}
		if (laser_number == LASER_2)
		{
			plane->A = 0;
			plane->B = -params.compl_inclination_coeff;
			plane->C = 1;
		}
		if (laser_number == VERTICAL_LINE)
		{
			plane->A = 0;
			plane->B = 1;
			plane->C = 0;
		}
	}
	plane->D = -plane->A * laser.x - plane->B * laser.y - plane->C * laser.z;
}

void fillSliceWithTriangles(const PolygonMesh &mesh, vector<int> *triangles_index, int laser_number, const SliceParams &slice_params, const SimulationParams &params)
{
	PointCloud<PointXYZ> mesh_vertices;
	PointXYZ point_1, point_2, point_3;
	int min_slice, max_slice;
	int slice_point_1, slice_point_2, slice_point_3;

	// Convert mesh in a point cloud (only vertices)
	fromPCLPointCloud2(mesh.cloud, mesh_vertices);

	for (int i = 0; i < mesh.polygons.size(); ++i)
	{
		point_1.x = mesh_vertices.points[mesh.polygons[i].vertices[0]].x;
		point_1.y = mesh_vertices.points[mesh.polygons[i].vertices[0]].y;
		point_1.z = mesh_vertices.points[mesh.polygons[i].vertices[0]].z;

		point_2.x = mesh_vertices.points[mesh.polygons[i].vertices[1]].x;
		point_2.y = mesh_vertices.points[mesh.polygons[i].vertices[1]].y;
		point_2.z = mesh_vertices.points[mesh.polygons[i].vertices[1]].z;

		point_3.x = mesh_vertices.points[mesh.polygons[i].vertices[2]].x;
		point_3.y = mesh_vertices.points[mesh.polygons[i].vertices[2]].y;
		point_3.z = mesh_vertices.points[mesh.polygons[i].vertices[2]].z;

		// Get slice indexes of the triangle vertices
		slice_point_1 = getSliceIndex(point_1, laser_number, slice_params, params);
		slice_point_2 = getSliceIndex(point_2, laser_number, slice_params, params);
		slice_point_3 = getSliceIndex(point_3, laser_number, slice_params, params);


		// Find vertex with lower slice index
		if (slice_point_1 < slice_point_2 && slice_point_1 < slice_point_3)
			min_slice = slice_point_1;
		else if (slice_point_2 < slice_point_3)
			min_slice = slice_point_2;
		else
			min_slice = slice_point_3;

		// Find vertex with upper slice index
		if (slice_point_1 > slice_point_2 && slice_point_1 > slice_point_3)
			max_slice = slice_point_1;
		else if (slice_point_2 > slice_point_3)
			max_slice = slice_point_2;
		else
			max_slice = slice_point_3;

		// Assign triangle to the correct slices
		if (min_slice != INDEX_NOT_FOUND && max_slice != INDEX_NOT_FOUND)
		{
			for (int z = min_slice; z <= max_slice; ++z)
				triangles_index[z].push_back(i);
		}
	}
}

void createTrianglesArray(const PolygonMesh &mesh, Triangle *triangles, vector<int> *triangles_index, int num_triangles_index_array)
{
	PointCloud<PointXYZ> mesh_vertices;

	// Convert mesh in a point cloud (only vertices)
	fromPCLPointCloud2(mesh.cloud, mesh_vertices);

	PointXYZ tmp;
	int count = 0;

	for (int i = 0; i < num_triangles_index_array; ++i)
	{
		for (int k = 0; k < triangles_index[i].size(); ++k)
		{
			tmp = mesh_vertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[0]];
			triangles[count].vertex_1.points[X] = tmp.x;
			triangles[count].vertex_1.points[Y] = tmp.y;
			triangles[count].vertex_1.points[Z] = tmp.z;

			tmp = mesh_vertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[1]];
			triangles[count].vertex_2.points[X] = tmp.x;
			triangles[count].vertex_2.points[Y] = tmp.y;
			triangles[count].vertex_2.points[Z] = tmp.z;

			tmp = mesh_vertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[2]];
			triangles[count].vertex_3.points[X] = tmp.x;
			triangles[count].vertex_3.points[Y] = tmp.y;
			triangles[count].vertex_3.points[Z] = tmp.z;

			count++;
		}
	}
}

void createSliceBoundArray(int *slice_bound, vector<int> *triangles_index, int *array_size)
{
	*array_size = 0;
	for (int i = 0; i < SLICE_NUMBER; ++i)
	{
		*array_size += triangles_index[i].size();
		slice_bound[i] = *array_size;
	}

	for (int i = SLICE_NUMBER; i < SLICE_NUMBER * 2; ++i)
	{
		*array_size += triangles_index[i].size();
		slice_bound[i] = *array_size;
	}

	for (int i = SLICE_NUMBER * 2; i < SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER; ++i)
	{
		*array_size += triangles_index[i].size();
		slice_bound[i] = *array_size;
	}
}

int getSliceIndex(const PointXYZ &laser_point, int laser_number, const SliceParams &slice_params, const SimulationParams &params)
{
	float slice_length = slice_params.slice_length * params.compl_inclination_coeff;
	float vertical_slice_length = slice_params.vertical_slice_length;

	if (laser_number == LASER_1)
	{
		const Plane &origin_plane = slice_params.origin_plane_laser1;
		for (int i = 0; i < SLICE_NUMBER; ++i)
		{
			if (origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D - slice_length < slice_length * i &&
				origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D >= slice_length * i)
			
				return i;
			
		}
	}
	if (laser_number == LASER_2)
	{
		const Plane &origin_plane = slice_params.origin_plane_laser2;
		for (int i = 0; i < SLICE_NUMBER; ++i)
		{
			if (origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D <= -slice_length * i &&
				origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D + slice_length > -slice_length * i)
			
				return i + SLICE_NUMBER;
			
		}
	}
	if (laser_number == VERTICAL_LINE)
	{
		for (int i = 0; i < VERTICAL_SLICE_NUMBER; ++i)
		{
			const Plane &origin_plane = slice_params.origin_vertical_plane;
			if (origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D - vertical_slice_length < vertical_slice_length * i &&
				origin_plane.A * laser_point.x + origin_plane.B * laser_point.y + origin_plane.C * laser_point.z + origin_plane.D >= vertical_slice_length * i)
		
				return i + SLICE_NUMBER * 2;
			
		}
	}

	return INDEX_NOT_FOUND;
}

void makeOptiziationSlice(PolygonMesh &mesh, const SliceParams &slice_params, const SimulationParams &params, int *slice_bound, Triangle **triangles_array, int *array_size)
{
	vector<int> *triangles_index = new vector<int>[SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER];

	// Slice for LASER 1
	fillSliceWithTriangles(mesh, triangles_index, LASER_1, slice_params, params);

	// Slice for LASER 2
	fillSliceWithTriangles(mesh, triangles_index, LASER_2, slice_params, params);

	// Vertical slice for occlusion optimization
	fillSliceWithTriangles(mesh, triangles_index, VERTICAL_LINE, slice_params, params);

	// Create slice bound array
	createSliceBoundArray(slice_bound, triangles_index, array_size);

	// Create triangles array used by OpenCL
	*triangles_array = new Triangle[*array_size];
	createTrianglesArray(mesh, *triangles_array, triangles_index, SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER);

	// Delete mesh and triangles index array from memory
	mesh.~PolygonMesh();
	delete[] triangles_index;

}

void initializeOpenCL(OpenCLDATA *data, Triangle *triangles_array, int array_lenght, int array_size_hits)
{
	cl_int err = CL_SUCCESS;

	try {
		// Check platforms available
		cl::Platform::get(&data->platforms);
		if (data->platforms.size() == 0)
		{
			cerr << endl << "ERRORE OpenCL: dimensione platform 0" << endl;
			exit(1);
		}
		
		// Choose first platform available
		for (int i = 0; i < data->platforms.size(); ++i)
		{
			bool catched = false;
			try
			{
				// Get list of devices on default platform and create context
				cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(data->platforms[i])(), 0 };
				data->context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
				data->devices = data->context.getInfo<CL_CONTEXT_DEVICES>();

				// Create command queue for first device
				data->queue = cl::CommandQueue(data->context, data->devices[0], 0, &err);
			}
			catch (cl::Error er)
			{
				catched = true;
			}
			if (!catched)
				break;
		}
		
		FILE *program_handle;
		size_t kernel_source_size;
		char *kernel_source;

		// Get size of kernel source
		int j = fopen_s(&program_handle, "intersection_opencl.cl", "rb");
		if (j != 0)
		{
			cerr << endl << "File kernel OpenCL non trovato" << endl;
			exit(1);
		}
		 
		fseek(program_handle, 0, SEEK_END);
		kernel_source_size = ftell(program_handle);
		rewind(program_handle);

		// Read kernel source into buffer
		kernel_source = (char*)malloc(kernel_source_size + 1);
		kernel_source[kernel_source_size] = '\0';
		fread(kernel_source, sizeof(char), kernel_source_size, program_handle);
		fclose(program_handle);

		// Build kernel from source string
		data->program_ = cl::Program(data->context, kernel_source);
		err = data->program_.build(data->devices);

		if (err != CL_SUCCESS)
		{
			cerr << "ERRORE OpenCL: compilazione kernel" << endl;
			exit(1);
		}
		
		free(kernel_source);

		// Size, in bytes, of each vector
		data->triangles_array_size = array_lenght * sizeof(Triangle);
		data->points_size = array_size_hits * sizeof(Vec3);
		data->hits_size = array_size_hits * sizeof(uchar);

		// Create device memory buffers
		data->device_triangles_array = cl::Buffer(data->context, CL_MEM_READ_ONLY, data->triangles_array_size);
		data->device_output_points = cl::Buffer(data->context, CL_MEM_WRITE_ONLY, data->points_size);
		data->device_output_hits = cl::Buffer(data->context, CL_MEM_WRITE_ONLY, data->hits_size);

		// Bind memory buffers
		data->queue.enqueueWriteBuffer(data->device_triangles_array, CL_TRUE, 0, data->triangles_array_size, triangles_array);
		data->queue.finish();

		// Delete triangles array (the same written in OpenCL buffer)
		delete[] triangles_array;

		// Create kernel object
		data->kernel = cl::Kernel(data->program_, "kernelTriangleIntersection", &err);

		// Bind kernel arguments to kernel
		err = data->kernel.setArg(0, data->device_triangles_array);
		err = data->kernel.setArg(1, data->device_output_points);
		err = data->kernel.setArg(2, data->device_output_hits);

		if (err != CL_SUCCESS)
		{
			cerr << endl << "ERRORE OpenCL: passaggio argomenti kernel" << endl;
			exit(1);
		}

	}

	catch (cl::Error er)
	{
		cerr << endl << "ERRORE OpenCL: inizializzazione" << endl;
		exit(1);
	}

}

void executeOpenCL(OpenCLDATA *data, Vec3 *output_points, uchar *output_hits, int start_index, int array_lenght, const Vec3 &ray_origin, const Vec3 &ray_direction) 
{
	cl_int err = CL_SUCCESS;

	err = data->kernel.setArg(3, start_index);
	err = data->kernel.setArg(4, array_lenght);
	err = data->kernel.setArg(5, ray_origin);
	err = data->kernel.setArg(6, ray_direction);

	if (err != CL_SUCCESS)
	{
		cerr << endl << "ERRORE OpenCL: passaggio argomenti kernel" << endl;
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

	// Block until kernel completation
	event.wait();

	// Read back device_output_point, device_output_hit
	data->queue.enqueueReadBuffer(data->device_output_points, CL_TRUE, 0, data->points_size, output_points);
	data->queue.enqueueReadBuffer(data->device_output_hits, CL_TRUE, 0, data->hits_size, output_hits);
}

void getIntersectionPoints(OpenCLDATA *data, Vec3 *output_points, uchar *output_hits, const PointXYZ &laser_point, const SimulationParams &params, const SliceParams &slice_params,
	PointCloud<PointXYZRGB>::Ptr cloud_intersection, const int laser_number, const int *slice_bound)
{
	float ray_density = (params.aperture_coefficient * 2) / params.number_of_line;

	char d_1 = 0;
	char d_2 = 0;
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		d_1 = 0;
		d_2 = 1;
	}

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		d_1 = 1;
		d_2 = 0;
	}


	int lower_bound, upper_bound;

	int k = getSliceIndex(laser_point, laser_number, slice_params, params);
	if (k < 0)
		return;

	if (k == 0)
		lower_bound = 0;
	else
		lower_bound = slice_bound[k-1];
	
	upper_bound = slice_bound[k];

	// Amount of triangles to find intersection with
	int diff = upper_bound - lower_bound;

	PointXYZRGB higher_intersection;
	Vec3 ray_origin, ray_direction;

	ray_origin.points[X] = laser_point.x;
	ray_origin.points[Y] = laser_point.y;
	ray_origin.points[Z] = laser_point.z;
	
	if (diff > 0)
	{
		// Cycle that examines the lines of the same laser beam
		for (int j = 0; j < params.number_of_line; ++j)
		{
			higher_intersection.z = VTK_FLOAT_MIN;

			// Set line direction
			ray_direction.points[d_1] = -params.aperture_coefficient + j * ray_density;
			ray_direction.points[d_2] = laser_number * params.inclination_coefficient;
			ray_direction.points[Z] = -1;

			// Start calculation of intersections with OpenCL
			executeOpenCL(data, output_points, output_hits, lower_bound, diff, ray_origin, ray_direction);

			// Look for intersection point with higher Z
			int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
			for (int h = 0; h < n_max; ++h)
			{
				if (output_hits[h] == HIT)
				{
					if (output_points[h].points[Z] >= higher_intersection.z)
					{
						higher_intersection.x = output_points[h].points[X];
						higher_intersection.y = output_points[h].points[Y];
						higher_intersection.z = output_points[h].points[Z];
						higher_intersection.r = 255;
						higher_intersection.g = 0;
						higher_intersection.b = 0;
					}
				}
			}
			if (higher_intersection.z > VTK_FLOAT_MIN)
				cloud_intersection->push_back(higher_intersection);
		}
	}
}

bool isOccluded(const PointXYZRGB &point, const PointXYZ &pin_hole, OpenCLDATA *data, const SliceParams &slice_params, const SimulationParams &params, 
	const int *slice_bound, Vec3 *output_points, uchar *output_hits)
{
	int slice_of_point, slice_of_pinhole;
	int lower_bound, upper_bound;
	PointXYZ point_to_check;

	point_to_check.x = point.x;
	point_to_check.y = point.y;
	point_to_check.z = point.z;

	// Get slices of point to check and pin hole
	slice_of_point = getSliceIndex(point_to_check, VERTICAL_LINE, slice_params, params);
	slice_of_pinhole = getSliceIndex(pin_hole, VERTICAL_LINE, slice_params, params);

	if (slice_of_point == INDEX_NOT_FOUND || slice_of_pinhole == INDEX_NOT_FOUND)
		return true;
	
	// Set lower and upper bound due to slices found
	if (slice_of_point < slice_of_pinhole)
	{
		if (slice_of_point == 0)
			lower_bound = 0;
		else
			lower_bound = slice_bound[slice_of_point - 1];

		upper_bound = slice_bound[slice_of_pinhole];
	}
	else
	{
		if (slice_of_pinhole == 0)
			lower_bound = 0;
		else
			lower_bound = slice_bound[slice_of_pinhole - 1];

		upper_bound = slice_bound[slice_of_point];
	}

	// Amount of triangles to find intersection with
	int diff = upper_bound - lower_bound;

	Vec3 origin;
	origin.points[X] = point.x;
	origin.points[Y] = point.y;
	origin.points[Z] = point.z;

	Vec3 direction;
	direction.points[X] = pin_hole.x - point.x;
	direction.points[Y] = pin_hole.y - point.y;
	direction.points[Z] = pin_hole.z - point.z;

	if (diff > 0)
	{
		PointXYZ higher_intersection;
		higher_intersection.z = VTK_FLOAT_MIN;

		executeOpenCL(data, output_points, output_hits, lower_bound, diff, origin, direction);
		int n_max = (int)(ceil((diff / (float)RUN) / LOCAL_SIZE) * LOCAL_SIZE);
		
		// Look for intersection point with higher Z
		for (int h = 0; h < n_max; ++h)
		{
			if (output_hits[h] == HIT)
			{
				if (output_points[h].points[Z] >= higher_intersection.z)
				{
					higher_intersection.x = output_points[h].points[X];
					higher_intersection.y = output_points[h].points[Y];
					higher_intersection.z = output_points[h].points[Z];
				}
			}
		}
		// If point obtained is very close to the point to check, then they are the same point,
		// else are different points. In this case the point to check is really occluded.
		if (higher_intersection.z > VTK_FLOAT_MIN)
			if (pointsDistance(higher_intersection, point_to_check) > EPSILON_OCCLUSION)
				return true;
	}

	return false;
}

void cameraSnapshot(const Camera &camera, const PointXYZ &pin_hole, const PointXYZ &laser_1, const PointXYZ &laser_2, PointCloud<PointXYZRGB>::Ptr cloud_intersection, Mat *img, 
	const SimulationParams &params, OpenCLDATA *data, Vec3 *output_points, const SliceParams &slice_params, const int *slice_bound, uchar *output_hits)
{
	// Initialize a white image
	*img = Mat::zeros(camera.image_height, camera.image_width, CV_8UC3);

	PointCloud<PointXYZ>::Ptr cloud_source(new PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr cloud_target(new PointCloud<PointXYZ>);

	cloud_source->push_back(pin_hole);
	cloud_source->push_back(laser_1);
	cloud_source->push_back(laser_2);

	PointXYZ c, p_1, p_2;

	// Camera
	c.x = c.y = c.z = 0;
	
	// Laser 1
	p_1.x = p_1.z = 0;
	p_1.y = -params.baseline;

	// Laser 2
	p_2.x = p_2.z = 0;
	p_2.y = params.baseline;

	cloud_target->push_back(c);
	cloud_target->push_back(p_1);
	cloud_target->push_back(p_2);

	// Calculate trasformation matrix
	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>  trans_est;
	registration::TransformationEstimationSVD<PointXYZ, PointXYZ>::Matrix4 transform_mat;
	trans_est.estimateRigidTransformation(*cloud_source, *cloud_target, transform_mat);

	// Points to project
	vector<Point3d> points;
	vector<Point2d> output_point;
	Eigen::Vector4f v_point, v_point_final;

	for (int i = 0; i < cloud_intersection->size(); ++i)
	{
		
		v_point[0] = cloud_intersection->points[i].x;
		v_point[1] = cloud_intersection->points[i].y;
		v_point[2] = cloud_intersection->points[i].z;
		v_point[3] = 1;
		v_point_final = transform_mat * v_point;

		v_point_final[2] = -v_point_final[2];

		points.push_back(Point3f(v_point_final[0], v_point_final[1], v_point_final[2]));
	}

	if (cloud_intersection->size() > 0)
	{
		projectPoints(points, Mat::zeros(3, 1, CV_64F), Mat::zeros(3, 1, CV_64F), camera.camera_matrix, camera.distortion, output_point);
		Point2d pixel;
		for (int i = 0; i < output_point.size(); ++i)
		{
			pixel = output_point.at(i);

			// Add 0.5 for a correct rounding
			pixel.x += 0.5;
			pixel.y += 0.5;

			if ((pixel.y >= 0) && (pixel.y < img->rows) && (pixel.x >= 0) && (pixel.x < img->cols))
			{
				// Check if point is occluded
				if ( !(isOccluded(cloud_intersection->at(i), pin_hole, data, slice_params, params,
					slice_bound, output_points, output_hits)) )
				{
					img->at<Vec3b>((int)(pixel.y), (int)(pixel.x))[0] = 0;
					img->at<Vec3b>((int)(pixel.y), (int)(pixel.x))[1] = 0;
					img->at<Vec3b>((int)(pixel.y), (int)(pixel.x))[2] = 255;
				}
			}
		}
	}
}

void imageToCloud(Camera &camera, const SimulationParams &params, const PointXYZ &laser_1, const PointXYZ &laser_2, const PointXYZ &pin_hole, Mat *image, PointCloud<PointXYZ>::Ptr cloud_out)
{
	// The point to add at the cloud
	PointXYZ point;	

	// Directional vector for the line pin_hole - point in the sensor
	float dx, dy, dz;			

	// Origin of the sensor in the space
	float x_sensor_origin = 0;
	float y_sensor_origin = 0;	

	// Sensor - pin hole offset
	float delta_x = ((image->cols / 2) - camera.camera_matrix.at<double>(0, 2)) * camera.pixel_dimension;
	float delta_y = ((image->rows / 2) - camera.camera_matrix.at<double>(1, 2)) * camera.pixel_dimension;

	// Calculate focal length
	float focal_length_x = camera.camera_matrix.at<double>(0, 0) * camera.pixel_dimension;
	float focal_length_y = camera.camera_matrix.at<double>(1, 1) * camera.pixel_dimension;
	float focal_length = (focal_length_x + focal_length_y) / 2;

	// Sensor translation
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

	// Undistort the image accord with the camera disortion parameters
	if (params.undistortion_flag)
	{
		Mat image_undistort;
		undistort(*image, image_undistort, camera.camera_matrix, camera.distortion);
		flip(image_undistort, *image, 0);
	}
	
	else
		flip(*image, *image, 0);

	// Calculate laser planes
	Plane plane_1, plane_2;
	getPlaneCoefficents(laser_1, &plane_1, LASER_1, params);
	getPlaneCoefficents(laser_2, &plane_2, LASER_2, params);


	// Project the ROI 1 points on the first plane
	for (int j = 0; j < image->cols; ++j)
	{
		for (int i = params.roi_1_start; i < params.roi_1_start + params.roi_dimension; ++i)
		{
			Vec3b &color = image->at<Vec3b>(i, j);

			// Check the color of the pixels
			if (color[2] != 0)
			{

				// Put the points of the image in the virtual sensor in the space
				if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
				{
					point.x = i * camera.pixel_dimension + x_sensor_origin;
					point.y = j * camera.pixel_dimension + y_sensor_origin;
				}

				if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
				{
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
	for (int j = 0; j < image->cols; ++j)
	{
		for (int i = params.roi_2_start; i < params.roi_2_start + params.roi_dimension; ++i)
		{
			Vec3b & color = image->at<Vec3b>(i, j);
			// Check the color of the pixels
			if (color[2] != 0)
			{
				// Put the points of the image on the virtual sensor in the space
				if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
				{
					point.x = i * camera.pixel_dimension + x_sensor_origin;
					point.y = j * camera.pixel_dimension + y_sensor_origin;
				}

				if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
				{
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

void loadMesh(const string path_file, PolygonMesh *mesh)
{
	// As reported in "vtk_lib_io.cpp" this fuction doesn't let
	// manage exception thrown by vtk readers.
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
		catch (IOException)
		{
			PCL_ERROR("Failed to save PCD file\n");
		}

	}
	else
		cerr << "WARNING! Point Cloud vuota" << endl;
}

string printProgBar(int percent) 
{
	stringstream prog;
	string bar;

	for (int i = 0; i < 50; ++i) {
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
	const MeshBounds &bounds, float *increment, float *current_position, int *number_of_iterations, float *final_pos)
{
	// Step size (in mm) between two snapshots
	*increment = params.scan_speed / camera.fps;

	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		*current_position = pin_hole.x;
		*final_pos = bounds.max_x + (bounds.min_x - laser_origin_2.x);
		*number_of_iterations = (int) ((*final_pos - laser_origin_1.x) / *increment);
	}
	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		*current_position = pin_hole.y;
		*final_pos = bounds.max_y + (bounds.min_y - laser_origin_2.y);
		*number_of_iterations = (int) ((*final_pos - laser_origin_1.y) / *increment);
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

void visualizeCloud(PointCloud<PointXYZ>::Ptr cloud)
{
	visualization::PCLVisualizer viewer("Viewer");
	viewer.addCoordinateSystem(100, "Viewer");
	viewer.addPointCloud<PointXYZ>(cloud, "Cloud");
	viewer.spin();
}