/*
* LaserSimulator
* Created on: 02/02/2016
* Last Update: 21/04/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/


#include "laser_scan_simulator_functions.h"
#include <iostream>


int intermediatePlanesPoints(const PolygonMesh &mesh, const Plane &plane, float slice_length, const SimulationParams &params, int laser_number, vector<int> *triangles_index)
{
	slice_length = slice_length *tan(deg2rad(params.laser_inclination));
	PointCloud<PointXYZ> cloud_mesh;
	PointXYZ point;

	// Convert mesh in a point cloud (only vertex)
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// Search minimum and maximum points on X, Y and Z axis
	for (int i = 0; i < mesh.polygons.size(); i++)
	{
		for (int j = 0; j <= 2; j++)
		{
			point.x = cloud_mesh.points[mesh.polygons[i].vertices[j]].x;
			point.y = cloud_mesh.points[mesh.polygons[i].vertices[j]].y;
			point.z = cloud_mesh.points[mesh.polygons[i].vertices[j]].z;

			if (laser_number == LASER_1 &&
				plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D - slice_length < 0 &&
				plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D > 0)
			{
				triangles_index->push_back(i);
				break;
			}

			if (laser_number == LASER_2 &&
				plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D < 0 &&
				plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D + slice_length> 0)
			{
				triangles_index->push_back(i);
				break;
			}
		}
	}
	return triangles_index->size();
}

int findMaxPoint(PointXYZ point_1, PointXYZ point_2, PointXYZ point_3, const SimulationParams &params)
{
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		if (point_1.x > point_2.x && point_1.x > point_3.x)
			return 0;
		else
		{
			if (point_2.x > point_3.x)
				return 1;
			else
				return 2;
		}
	}

	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		if (point_1.y > point_2.y && point_1.y > point_3.y)
			return 0;
		else
		{
			if (point_2.y > point_3.y)
				return 1;
			else
				return 2;
		}
	}
}

int findMinPoint(PointXYZ point_1, PointXYZ point_2, PointXYZ point_3, const SimulationParams &params)
{
	if (params.scan_direction == DIRECTION_SCAN_AXIS_X)
	{
		if (point_1.x < point_2.x && point_1.x < point_3.x)
			return 0;
		else
		{
			if (point_2.x < point_3.x)
				return 1;
			else
				return 2;
		}
	}

	if (params.scan_direction == DIRECTION_SCAN_AXIS_Y)
	{
		if (point_1.y < point_2.y && point_1.y < point_3.y)
			return 0;
		else
		{
			if (point_2.y < point_3.y)
				return 1;
			else
				return 2;
		}
	}
}

int fillSliceWithTriangles(PolygonMesh mesh, vector<int> *triangles_index, const Plane &origin_plane, int laser_number, float slice_length, const int slice_number, const SimulationParams &params)
{
	PointCloud<PointXYZ> cloud_mesh;
	PointXYZ point1, point2, point3, min_point, max_point;

	int lost_triangle = 0;
	// Convert mesh in a point cloud (only vertex)
	fromPCLPointCloud2(mesh.cloud, cloud_mesh);

	// Search minimum and maximum points on X, Y and Z axis
	for (int i = 0; i < mesh.polygons.size(); i++)
	{
		point1.x = cloud_mesh.points[mesh.polygons[i].vertices[0]].x;
		point1.y = cloud_mesh.points[mesh.polygons[i].vertices[0]].y;
		point1.z = cloud_mesh.points[mesh.polygons[i].vertices[0]].z;

		point2.x = cloud_mesh.points[mesh.polygons[i].vertices[1]].x;
		point2.y = cloud_mesh.points[mesh.polygons[i].vertices[1]].y;
		point2.z = cloud_mesh.points[mesh.polygons[i].vertices[1]].z;

		point3.x = cloud_mesh.points[mesh.polygons[i].vertices[2]].x;
		point3.y = cloud_mesh.points[mesh.polygons[i].vertices[2]].y;
		point3.z = cloud_mesh.points[mesh.polygons[i].vertices[2]].z;

		// Cerco quali sono i punti minimi e massimi del triangolo
		int min_point_index = findMinPoint(point1, point2, point3, params);
		int max_point_index = findMaxPoint(point1, point2, point3, params);

		// Salvo i punti minimi e massimi del triangolo
		min_point.x = cloud_mesh.points[mesh.polygons[i].vertices[min_point_index]].x;
		min_point.y = cloud_mesh.points[mesh.polygons[i].vertices[min_point_index]].y;
		min_point.z = cloud_mesh.points[mesh.polygons[i].vertices[min_point_index]].z;

		max_point.x = cloud_mesh.points[mesh.polygons[i].vertices[max_point_index]].x;
		max_point.y = cloud_mesh.points[mesh.polygons[i].vertices[max_point_index]].y;
		max_point.z = cloud_mesh.points[mesh.polygons[i].vertices[max_point_index]].z;

		// Guardo questi valori in che fetta devono finire
		int min_slice = getSliceIndex(min_point, origin_plane, laser_number, slice_length, slice_number, params);
		int max_slice = getSliceIndex(max_point, origin_plane, laser_number, slice_length, slice_number, params);

		// Assegno il triango alle fette corrispondenti
		if (min_slice != -1 && max_slice != -1)
		{
			for (int z = min_slice; z <= max_slice; z++)
			{
				triangles_index[z].push_back(i);
			}
		}
		else
		{
			lost_triangle++;
		}
	}
	return lost_triangle;
}

void createAllTriangleArray2(const PolygonMesh &mesh, Triangle* triangles, vector<int> *triangles_index, int num_triangles_index_array)
{
	PointCloud<PointXYZ> meshVertices;
	fromPCLPointCloud2(mesh.cloud, meshVertices);

	PointXYZ tmp;
	int count = 0;
	for (int i = 0; i < num_triangles_index_array; i++)
	{
		for (int k = 0; k < triangles_index[i].size(); k++)
		{
			tmp = meshVertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[0]];
			triangles[count].vertex_1.points[X] = tmp.x;
			triangles[count].vertex_1.points[Y] = tmp.y;
			triangles[count].vertex_1.points[Z] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[1]];
			triangles[count].vertex_2.points[X] = tmp.x;
			triangles[count].vertex_2.points[Y] = tmp.y;
			triangles[count].vertex_2.points[Z] = tmp.z;

			tmp = meshVertices.points[mesh.polygons[triangles_index[i].at(k)].vertices[2]];
			triangles[count].vertex_3.points[X] = tmp.x;
			triangles[count].vertex_3.points[Y] = tmp.y;
			triangles[count].vertex_3.points[Z] = tmp.z;
			count++;
		}
	}
}

void createSliceBoundArray(int *slice_bound, vector<int> *triangles_index, int * total_triangle, int slice_number)
{
	for (int i = 0; i < slice_number; i++)
	{
		*total_triangle += triangles_index[i].size();
		slice_bound[i] = *total_triangle;
		cout << "Numero triangoli nella fetta " << i << " = " << triangles_index[i].size() << endl;
		//int index = getSliceIndex(laser_origin_1, origin_plane_laser1, LASER_1, slice_length, slice_number, params);
		//laser_origin_1.y += slice_length;
		//cout << "L'indice si trova nella fetta " << index << endl;
	}

	for (int i = slice_number; i < slice_number * 2; i++)
	{
		*total_triangle += triangles_index[i].size();
		slice_bound[i] = *total_triangle;
		cout << "Numero triangoli nella fetta " << i << " = " << triangles_index[i].size() << endl;
		//int index = getSliceIndex(laser_origin_2, origin_plane_laser2, LASER_2, slice_length, slice_number, params);
		//laser_origin_2.y += slice_length;
		//cout << "L'indice si trova nella fetta " << index << endl;
	}
}

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


	/********* Read data from XML parameters file ****************************/
	readParamsFromXML(&camera, &params, &snapshot_save_flag, &path_read_file, &path_save_file);

	// Starting time counter
	high_resolution_clock::time_point start = high_resolution_clock::now();


	/*************************** Load mesh ***********************************/
	loadMesh(path_read_file, &mesh);
	cout << "Dimensione della mesh (n. triangoli): " << mesh.polygons.size() << endl << endl;

	// Arrays used to optimize the computation of intersections
	float *max_point_triangle = new float[mesh.polygons.size()];
	int *max_point_triangle_index = new int[mesh.polygons.size()];


	/************* Find minimum and mixiumum points of 3 axis and fill ********************/
	/************ arrays used to find maximum value on the direction axis *****************/
	calculateBoundariesAndArrayMax(params, mesh, max_point_triangle_index, max_point_triangle, &bounds);


	/********** Print minimum and maximum points of mesh *********************/
	cout << "Estremi della mesh:" << endl << getMeshBoundsValues(bounds);

	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	// INIZIO AFFETTATURA
	int slice_number = 50;
	float slice_length = (bounds.max_y - laser_origin_1.y) / slice_number;
	vector<int> *triangles_index = new vector<int>[slice_number * 2];
	int *slice_bound = new int[slice_number * 2];
	int total_triangle = 0;

	/*********************** Find "big" triangles ****************************/
	vector<Triangle> big_triangles_vec;
	vector<int> big_triangles_index;
	//findBigTriangles(mesh, bounds, params, &big_triangles_vec, &big_triangles_index, mesh.polygons.size(), slice_length);

	// Put big triangles in a Triangle array
	int big_array_size = big_triangles_vec.size();
	Triangle *big_triangles = new Triangle[big_array_size];
	for (int i = 0; i < big_array_size; i++)
		big_triangles[i] = big_triangles_vec[i];
	cout << "Numero triangoli \"grandi\": " << big_array_size << endl << endl;
	/***************************************************************************/

	Plane origin_plane_laser1, origin_plane_laser2;
	origin_plane_laser1.A = 0;
	origin_plane_laser1.B = tan(deg2rad(params.laser_inclination));//1.73205081;
	origin_plane_laser1.C = 1;
	origin_plane_laser1.D = -origin_plane_laser1.A * laser_origin_1.x - origin_plane_laser1.B * laser_origin_1.y - origin_plane_laser1.C * laser_origin_1.z;

	origin_plane_laser2.A = 0;
	origin_plane_laser2.B = -tan(deg2rad(params.laser_inclination));//1.73205081;
	origin_plane_laser2.C = 1;
	origin_plane_laser2.D = -origin_plane_laser2.A * laser_origin_2.x - origin_plane_laser2.B * laser_origin_2.y - origin_plane_laser2.C * laser_origin_2.z;

	Plane plane_1, plane_2;

	int lost_triangle;
	// Affetta per il LASER 1
	lost_triangle = fillSliceWithTriangles(mesh, triangles_index, origin_plane_laser1, LASER_1, slice_length, slice_number, params);
	// Affetta per il LASER 2
	lost_triangle += fillSliceWithTriangles(mesh, triangles_index, origin_plane_laser2, LASER_2, slice_length, slice_number, params);


	// Costruisco l'array di bound
	createSliceBoundArray(slice_bound, triangles_index, &total_triangle, slice_number);

	cout << "LOST TRIANGLE: " << lost_triangle << endl;
	cout << "TOTAL TRIANGLE: " << total_triangle << endl;

	Triangle *array_laser = new Triangle[total_triangle];
	createAllTriangleArray2(mesh, array_laser, triangles_index, slice_number * 2);
	
	int array_size = total_triangle; 
	


	int array_size_hits = (int) (ceil(array_size / (float)RUN));
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	initializeOpenCL(&data, array_laser, array_size, big_triangles, big_array_size, array_size_hits);


	/**************** Set initial position for camera and lasers *****************/
	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	// Set current position, calculate final position and number of iterations
	float increment, current_position, number_of_iterations, final_pos;
	getScanCycleParams(params, camera, pin_hole, laser_origin_1, laser_origin_2, bounds, &increment, &current_position, &number_of_iterations, &final_pos);

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection_backup(new PointCloud<PointXYZRGB>);
	Mat image;
	

	/****************************************************************************************/
	/************ CORE OF THE PROJECT: this cycle simulates the laser scan. *****************/
	/******** In every iteration finds intersection with mesh, take a camera snapshot *******/
	/******** and reconstruct the points in the 3D space ************************************/
	/****************************************************************************************/
	for (int z = 0; (current_position - params.baseline) < final_pos; z++)
	{
		// Print progression bar and number of iteration completed
		cout << printProgBar((int) ((z / number_of_iterations) * 100 + 0.5));
		cout << z << " di " << (int) (number_of_iterations);

		// Update position of pin hole and lasers
		setLasersAndPinHole(&pin_hole, &laser_origin_1, &laser_origin_2, current_position, params);
		current_position += increment;


		/******************************* Look for intersection with mesh **************************************/
		// For laser 1
		getIntersectionOpenCL(&data, output_points, output_hits, mesh, laser_origin_1, params, cloud_intersection, origin_plane_laser1,
			&plane_1, LASER_1, bounds, array_size, big_array_size, slice_length, slice_number, slice_bound);
		// For laser 2
		getIntersectionOpenCL(&data, output_points, output_hits, mesh, laser_origin_2, params, cloud_intersection, origin_plane_laser2,
			&plane_2, LASER_2, bounds, array_size, big_array_size, slice_length, slice_number, slice_bound);


		/************************************ Take snapshot  **************************************************/
		cameraSnapshot(camera, pin_hole, laser_origin_1, laser_origin_2, cloud_intersection, &image, params, array_size, &data, output_points,
			output_hits, max_point_triangle);

		// Save snapshot (only for debug) 
		if (snapshot_save_flag)
			imwrite("../imgOut/out_" + to_string(z) + ".png", image);

		/*************************** Convert image to point cloud *********************************************/
		imageToCloud(camera, params, plane_1, plane_2, pin_hole, &image, cloud_out);


		// Make a backup of point cloud that contains (all) intersections
		for (int i = 0; i < cloud_intersection->size(); i++)
			cloud_intersection_backup->push_back(cloud_intersection->at(i));

		// Clear current intersections
		cloud_intersection->clear();
	}


	// Points of the cloud 
	cout << endl << endl << "Numero di punti cloud rossa opencl: " << cloud_intersection_backup->points.size() << endl;
	cout << endl << "Numero di punti della point cloud: " << cloud_out->points.size() << endl;


	// Save cloud 
	if (cloud_intersection_backup->size() > 0)
	{
		if (io::savePCDFileASCII("../result/all_intersection_cloud.pcd", *cloud_intersection_backup))
			PCL_ERROR("Failed to save PCD file\n");
	}
	else
		cerr << "WARNING! Point Cloud intersection is empty" << endl;

	saveCloud(path_save_file, cloud_out);


	/***************************** Visualize cloud ***************************************/
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