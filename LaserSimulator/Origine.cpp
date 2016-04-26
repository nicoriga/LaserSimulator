/*
* LaserSimulator
* Created on: 02/02/2016
* Last Update: 21/04/2016
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/


#include "laser_scan_simulator_functions.h"

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
	sortArrays(max_point_triangle, max_point_triangle_index, array_size);

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