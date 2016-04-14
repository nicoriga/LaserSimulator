#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace pcl;

int main(int argc, char** argv)
{

	PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);
	io::loadPCDFile("../dataset/prodottooccluso.pcd", *cloud_out);

	// Create a PCLVisualizer
	visualization::PCLVisualizer viewer("viewer");
	viewer.addCoordinateSystem(100, "viewer");
	viewer.addPointCloud<PointXYZ>(cloud_out, "cloudGen");
	viewer.spin();
	return 0;

}