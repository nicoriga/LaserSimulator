/*
* LaserSimulator
* Created on: 18/02/2016
* Last Update: 07/05/2016
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
	PointXYZ laser_origin_1, laser_origin_2, pin_hole;


	/********* Read data from parameters XML file ****************************/
	readParamsFromXML(&camera, &params, &snapshot_save_flag, &path_read_file, &path_save_file);


	// Starting time counter
	high_resolution_clock::time_point start = high_resolution_clock::now();


	/*************************** Load mesh ***********************************/
	loadMesh(path_read_file, &mesh);
	cout << "Dimensione della mesh (n. triangoli): " << mesh.polygons.size() << endl << endl;


	/*********** Find minimum and mixiumum points of the 3 axis ***************/
	calculateBoundaries(mesh, &bounds);

	// Print minimum and maximum points of mesh
	cout << "Estremi della mesh:" << endl << getMeshBoundsValues(bounds);

	// Set initial position of camera - laser system
	setInitialPosition(&pin_hole, &laser_origin_1, &laser_origin_2, params, bounds);


	/********************** Start slicing optimization ************************/
	cout << "Pre-elaborazione... ";

	// Set the parameter for the slice optimization
	setSliceParams(&slice_params, laser_origin_1, laser_origin_2, params, bounds);

	int *slice_bound = new int[SLICE_NUMBER * 2 + VERTICAL_SLICE_NUMBER];
	Triangle *triangles_array;
	int array_size;
	makeOptiziationSlice(mesh, slice_params, params, slice_bound, &triangles_array, &array_size);


	/********************** Inititialize OpenCL ********************************/
	int array_size_hits = (int) (ceil(array_size / (float)RUN));
	Vec3* output_points = new Vec3[array_size_hits];
	uchar* output_hits = new uchar[array_size_hits];
	initializeOpenCL(&data, triangles_array, array_size, array_size_hits);

	cout << "terminata, durata: " << returnTime(high_resolution_clock::now() - start) << endl;
	
	/********************** Start elaboration ************************/
	cout << endl << "Inizio elaborazione..." << endl;

	// Set current position, calculate final position and number of iterations
	float increment, current_position, final_pos;
	int number_of_iterations;
	getScanCycleParams(params, camera, pin_hole, laser_origin_1, laser_origin_2, bounds, &increment, &current_position, &number_of_iterations, &final_pos);

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection(new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZRGB>::Ptr cloud_intersection_backup(new PointCloud<PointXYZRGB>);
	Mat image;

	/****************************************************************************************/
	/******** CORE OF THE PROJECT: this cycle simulates the laser scan. *********************/
	/******** In every iteration finds intersection with mesh, take a camera snapshot *******/
	/******** and reconstruct the points in the 3D space ************************************/
	/****************************************************************************************/
	for (int z = 0; z <= number_of_iterations; ++z)
	{
		// Print progression bar and number of iteration completed
		cout << printProgBar((int)(z / (float)number_of_iterations * 100)) << z << " di " << number_of_iterations;

		// Update position of pin hole and lasers
		setLasersAndPinHole(&pin_hole, &laser_origin_1, &laser_origin_2, current_position, params);
		current_position += increment;


		/******************************* Look for intersections with mesh **************************************/
		// For laser 1
		getIntersectionPoints(&data, output_points, output_hits, laser_origin_1, params, slice_params, cloud_intersection, LASER_1, slice_bound);
		// For laser 2
		getIntersectionPoints(&data, output_points, output_hits, laser_origin_2, params, slice_params, cloud_intersection, LASER_2, slice_bound);


		/************************************ Take snapshot  **************************************************/
		cameraSnapshot(camera, pin_hole, laser_origin_1, laser_origin_2, cloud_intersection, &image, params, &data, output_points,
			slice_params, slice_bound, output_hits);

		// Save snapshot (only for debug) 
		if (snapshot_save_flag)
			imwrite("../imgOut/out_" + to_string(z) + ".png", image);


		/*************************** Convert image to point cloud *********************************************/
		imageToCloud(camera, params, laser_origin_1, laser_origin_2, pin_hole, &image, cloud_out);


		// Make a backup of point cloud that contains (all) intersections
		//for (int i = 0; i < cloud_intersection->size(); i++)
		//	cloud_intersection_backup->push_back(cloud_intersection->at(i));

		// Clear current intersections
		cloud_intersection->clear();
	}


	// Points of cloud 
	//cout << endl << endl << "Numero di punti cloud rossa opencl: " << cloud_intersection_backup->points.size() << endl;
	cout << endl << "Numero di punti della point cloud: " << cloud_out->points.size() << endl;

	// Print total time of computation 
	cout << endl << "Durata totale: " << returnTime(high_resolution_clock::now() - start) << endl;

	// Save cloud 
	saveCloud(path_save_file, cloud_out);

	/***************************** Visualize cloud ***************************************/
	visualizeCloud(cloud_out);




	return 0;
}