#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace pcl;

int main(int argc, char** argv)
{

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	PointCloud<PointXYZRGB>::Ptr cloud_test(new PointCloud<PointXYZRGB>);
	io::loadPCDFile("../LaserSimulator/all_intersection_cloud.pcd", *cloud_test);
	io::loadPCDFile("../LaserSimulator/final_cloud.pcd", *cloud_out);

/*	// Create a PCLVisualizer
	visualization::PCLVisualizer viewer("viewer");
	viewer.addCoordinateSystem(100, "viewer");
	viewer.addPointCloud<PointXYZ>(cloud_out, "cloudGen");
	viewer.spin();
*/

	visualization::PCLVisualizer viewer("PCL Viewer");
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud_test);

	int v1(0);
	//viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.0, 0.0, 1.0, 0.5, v1);
	viewer.addCoordinateSystem(100,"1", v1);
	viewer.addPointCloud<PointXYZRGB>(cloud_test, rgb, "ALL INTERSECTIONS", v1);

	int v2(0);
	//viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer.createViewPort(0.0, 0.5, 1.0, 1.0, v2);
	viewer.addCoordinateSystem(100,"2", v2);
	viewer.addPointCloud<PointXYZ>(cloud_out, "FINAL CLOUD", v2);

	viewer.spin();

	return 0;

}