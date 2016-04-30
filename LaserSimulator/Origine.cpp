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


	/********** Print minimum and maximum points of mesh *********************/
	cout << "Estremi della mesh:" << endl << getMeshBoundsValues(bounds);

	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	// INIZIO AFFETTATURA
	Plane origin_plane_laser1, origin_plane_laser2;

	int slice_number = 1000;
	float fp = bounds.max_y + (bounds.min_y - laser_origin_2.y);
	float slice_length = (fp - laser_origin_1.y) / slice_number;
	vector<int> *triangles_index = new vector<int>[slice_number * 2];
	int *slice_bound = new int[slice_number * 2];
	int total_triangle = 0;

	getPlaneCoefficents(laser_origin_1, &origin_plane_laser1, LASER_1, params);
	getPlaneCoefficents(laser_origin_2, &origin_plane_laser2 , LASER_2, params);

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
	createTrianglesArray(mesh, array_laser, triangles_index, slice_number * 2);
	
	int array_size = total_triangle;
	
	float *max_point_triangle;

	int array_size_hits = (int) (ceil(array_size / (float)RUN));
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	initializeOpenCL(&data, array_laser, array_size, array_size_hits);


	/**************** Set initial position for camera and lasers *****************/
	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);

	// Set current position, calculate final position and number of iterations
	float increment, current_position, number_of_iterations, final_pos;
	getScanCycleParams(params, camera, pin_hole, laser_origin_1, laser_origin_2, bounds, &increment, &current_position, &number_of_iterations, &final_pos);

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection_backup(new PointCloud<PointXYZRGB>);
	Mat image;
	Plane plane_1, plane_2;

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
			&plane_1, LASER_1, bounds, slice_length, slice_number, slice_bound);
		// For laser 2
		getIntersectionOpenCL(&data, output_points, output_hits, mesh, laser_origin_2, params, cloud_intersection, origin_plane_laser2,
			&plane_2, LASER_2, bounds, slice_length, slice_number, slice_bound);


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