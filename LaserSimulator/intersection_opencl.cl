/*
* Title: Laser scan simulator
* Created on: 18/02/2016
* Last Update: 09/06/2016
*
* Authors: Mauro Bagatella  1110345
*          Loris Del Monaco 1106940
*/

#define EPSILON 0.000001f
#define X 0
#define Y 1
#define Z 2

#define RUN 256
#define HIT 1
#define MISS 0

//#define TEST_CULL // Used to activate first branch of intersection algorithm

typedef struct 
{
	float points[3];	
} Vec3;


typedef struct 
{ 
	Vec3 vertex_1;
	Vec3 vertex_2;
	Vec3 vertex_3;
} Triangle;


// Vector dot product
float DOT(Vec3 a, Vec3 b)
{
	return a.points[X] * b.points[X] + a.points[Y] * b.points[Y] + a.points[Z] * b.points[Z];
}

// Vector subtraction
Vec3 SUB(Vec3 a, Vec3 b) 
{ 
	Vec3 c;
	c.points[X] = a.points[X] - b.points[X];
	c.points[Y] = a.points[Y] - b.points[Y];
	c.points[Z] = a.points[Z] - b.points[Z]; 
	return c;
}

// Vector addition
Vec3 ADD(Vec3 a, Vec3 b) 
{ 
	Vec3 c;
	c.points[X] = a.points[X] + b.points[X];
	c.points[Y] = a.points[Y] + b.points[Y];
	c.points[Z] = a.points[Z] + b.points[Z]; 
	return c;
}

// Vector cross product
Vec3 CROSS(Vec3 a, Vec3 b) 
{ 
	Vec3 c;
	c.points[X] = a.points[Y] * b.points[Z] - a.points[Z] * b.points[Y];
	c.points[Y] = a.points[Z] * b.points[X] - a.points[X] * b.points[Z];
	c.points[Z] = a.points[X] * b.points[Y] - a.points[Y] * b.points[X]; 
	return c;
}

// Vector scaling
Vec3 MUL(Vec3 a, float f)
{ 
	Vec3 b;
	b.points[X] = a.points[X] * f;
	b.points[Y] = a.points[Y] * f;
	b.points[Z] = a.points[Z] * f;
	return b;
}


// Moller-Trumbore intersection algorithm
inline int triangleIntersection(Vec3 V0, Vec3 V1, Vec3 V2, Vec3 O, Vec3 D, Vec3 *hit_point)
{
	Vec3 e1, e2;  //Edge1, Edge2
	Vec3 P, Q, T;
	float det, inv_det, u, v;
	float t;

	// Find vectors for two edges sharing V0
	e1 = SUB(V1, V0);
	e2 = SUB(V2, V0);

	// Begin calculating determinant - also used to calculate u parameter
	P = CROSS(D, e2);

	// if determinant is near zero, ray lies in plane of triangle
	det = DOT(e1, P);

#ifdef TEST_CULL
	// CULLING
	if (det < EPSILON)
		return MISS;

	// Calculate distance from V0 to ray origin
	T = SUB(O, V0);

	// Calculate u parameter and test bound
	u = DOT(T, P);

	// The intersection lies outside of the triangle
	if (u < 0.f || u > det)
		return MISS;

	// Prepare to test v parameter
	Q = CROSS(T, e1);

	// Calculate v parameter and test bound
	v = DOT(D, Q);

	// The intersection lies outside of the triangle
	if (v < 0.f || u + v > det)
		return MISS;

	t = DOT(e2, Q);

	inv_det = 1.f / det;

	t = t * inv_det;
	u = u * inv_det;
	v = v * inv_det;

	if (t > EPSILON) // Ray intersection
	{ 
		*hit_point = ADD(O, MUL(D, t));
		return HIT;
	}

#else
	// NOT CULLING
	if (det > -EPSILON && det < EPSILON)
		return MISS;
	
	inv_det = 1.f / det;

	// Calculate distance from V0 to ray origin
	T = SUB(O, V0);

	// Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;

	// The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f)
		return MISS;

	// Prepare to test v parameter
	Q = CROSS(T, e1);

	// Calculate v parameter and test bound
	v = DOT(D, Q) * inv_det;

	// The intersection lies outside of the triangle
	if (v < 0.f || u + v > 1.f)
		return MISS;

	t = DOT(e2, Q) * inv_det;

	if (t > EPSILON) // Ray intersection
	{ 
		*hit_point = ADD(O, MUL(D, t));
		return HIT;
	}

#endif

	return MISS;
}


__kernel void kernelTriangleIntersection(__global Triangle *input, __global Vec3 *output_point, __global uchar *output_hit,
										int start_index, int num_triangles, Vec3 ray_origin, Vec3 ray_direction)
{
	int k = get_global_id(0); 
	output_hit[k] = MISS;

	Vec3 local_hit_point, hight_hit_point;
	uchar local_hit = MISS;
	int j, t;
	int l = k * RUN + start_index;
	int final_index = num_triangles + start_index;

	for (j = 0; j < RUN && (l + j) < final_index; ++j)
	{
		t = l + j;

		if (triangleIntersection(input[t].vertex_1, input[t].vertex_2, input[t].vertex_3, ray_origin, ray_direction, &local_hit_point))
		{
			if (local_hit == MISS || local_hit_point.points[Z] >= hight_hit_point.points[Z])
			{
				local_hit = HIT;
					
				hight_hit_point.points[X] = local_hit_point.points[X];
				hight_hit_point.points[Y] = local_hit_point.points[Y];
				hight_hit_point.points[Z] = local_hit_point.points[Z];
			}
		}
	}

	if(local_hit == HIT)
	{
		output_hit[k] = HIT;
		output_point[k].points[X] = hight_hit_point.points[X];
		output_point[k].points[Y] = hight_hit_point.points[Y];
		output_point[k].points[Z] = hight_hit_point.points[Z];
	}
	
}
