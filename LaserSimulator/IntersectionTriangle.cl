struct Point3D{
	float x;
	float y;
	float z;
};
struct Triangle{ 
	Point3D vertex1;
	Point3D vertex2;
	Point3D vertex3;
};


__kernel void prefilter_norm(__global unsigned char *input, __global unsigned char *output,  int rows, int cols){
	int x = get_global_id(0);
	int y = get_global_id(1);
}