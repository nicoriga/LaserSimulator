/*
* LaserSimulator
* Created on: 02/02/2016
* Last Update: 21/04/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/


#include "laser_scan_simulator_functions.h"
#include <iostream>



int main(int argc, char** argv)
{
	PolygonMesh mesh;
	MeshBounds bounds;
	Camera camera;
	SimulationParams params;
	bool snapshot_save_flag;
	string path_read_file, path_save_file;
	OpenCLDATA data;
	SliceParams slice_params;

	// Origin point of laser 1, laser 2 and camera pin hole
	PointXYZ laser_origin_1, laser_origin_2, pin_hole;


	/********* Read data from XML parameters file ****************************/
	readParamsFromXML(&camera, &params, &snapshot_save_flag, &path_read_file, &path_save_file);

	// Starting time counter
	high_resolution_clock::time_point start = high_resolution_clock::now();


	/*************************** Load mesh ***********************************/
	loadMesh(path_read_file, &mesh);
	cout << "Dimensione della mesh (n. triangoli): " << mesh.polygons.size() << endl << endl;


	/************* Find minimum and mixiumum points of 3 axis *****************************/
	calculateBoundaries(params, mesh, &bounds);


	// Print minimum and maximum points of mesh
	cout << "Estremi della mesh:" << endl << getMeshBoundsValues(bounds);

	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	/************* Start slicing optimization *****************************/
	// Set the parameter for the slice optimization
	setSliceParams(&slice_params, laser_origin_1, laser_origin_2, params, bounds);
	
	vector<int> *triangles_index = new vector<int>[SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER];
	int *slice_bound = new int[SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER];
	int array_size = 0;
	int lost_triangle = 0;

	// Slice for LASER 1
	lost_triangle = fillSliceWithTriangles(mesh, triangles_index, slice_params.origin_plane_laser1, LASER_1, slice_params, params);
	// Slice for LASER 2
	lost_triangle += fillSliceWithTriangles(mesh, triangles_index, slice_params.origin_plane_laser2, LASER_2, slice_params, params);
	// Vertical slice for occlusion optimization
	lost_triangle += fillSliceWithTriangles(mesh, triangles_index, slice_params.origin_vertical_plane, VERTICAL_LINE, slice_params, params);

	// Create slice bound array
	createSliceBoundArray(slice_bound, triangles_index, &array_size);

	cout << "LOST TRIANGLE: " << lost_triangle << endl;
	cout << "TOTAL TRIANGLE: " << array_size << endl;

	// Create triangles array used by OpenCL
	Triangle *triangles_array = new Triangle[array_size];
	createTrianglesArray(mesh, triangles_array, triangles_index, SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER);
	
	// Delete mesh and triangles index array from memory
	mesh.~PolygonMesh();
	delete[] triangles_index;
	
	
	/**************** Inititialize OpenCL *****************/
	int array_size_hits = (int) (ceil(array_size / (float)RUN));
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	initializeOpenCL(&data, triangles_array, array_size, array_size_hits);

	// Delete array	written in OpenCL buffer
	delete[] triangles_array;


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
	for (int z = 0; z <= number_of_iterations; z++)
	{
		// Print progression bar and number of iteration completed
		cout << printProgBar((int) ((z / number_of_iterations) * 100 + 0.5));
		cout << z << " di " << (int) (number_of_iterations);

		// Update position of pin hole and lasers
		setLasersAndPinHole(&pin_hole, &laser_origin_1, &laser_origin_2, current_position, params);
		current_position += increment;

		//int slice_point1 = getSliceIndex(laser_origin_1, vertical_plane, VERTICAL_LINE, slice_length, vertical_slice_length, params);
		//cout << "Il pin hole � nella fetta " << slice_point1 - 2* SLICE_NUMBER << endl;
		/******************************* Look for intersection with mesh **************************************/
		// For laser 1
		getIntersectionOpenCL(&data, output_points, output_hits, laser_origin_1, params, slice_params, cloud_intersection, slice_params.origin_plane_laser1,
			LASER_1, bounds, slice_bound);
		// For laser 2
		getIntersectionOpenCL(&data, output_points, output_hits, laser_origin_2, params, slice_params, cloud_intersection, slice_params.origin_plane_laser2,
			LASER_2, bounds, slice_bound);


		/************************************ Take snapshot  **************************************************/
		cameraSnapshot(camera, pin_hole, laser_origin_1, laser_origin_2, cloud_intersection, &image, params, &data, output_points, slice_params.origin_vertical_plane, slice_params,
			slice_bound, output_hits);

		// Save snapshot (only for debug) 
		if (snapshot_save_flag)
			imwrite("../imgOut/out_" + to_string(z) + ".png", image);

		/*************************** Convert image to point cloud *********************************************/
		imageToCloud(camera, params, laser_origin_1, laser_origin_2, pin_hole, &image, cloud_out);


		// Make a backup of point cloud that contains (all) intersections
		for (int i = 0; i < cloud_intersection->size(); i++)
			cloud_intersection_backup->push_back(cloud_intersection->at(i));

		// Clear current intersections
		cloud_intersection->clear();
	}


	// Points of the cloud 
	cout << endl << endl << "Numero di punti cloud rossa opencl: " << cloud_intersection_backup->points.size() << endl;
	cout << endl << "Numero di punti della point cloud: " << cloud_out->points.size() << endl;

	// Print total time of computation 
	cout << endl << "Durata: " << returnTime(high_resolution_clock::now() - start) << endl;

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



	viewer.spin();


	return 0;
}