#define EPSILON 0.000001f
#define X 0
#define Y 1
#define Z 2

typedef struct 
{
	float points[3];	
} Vec3;

typedef struct 
{ 
	Vec3 vertex1;
	Vec3 vertex2;
	Vec3 vertex3;
} Triangle;

float DOT( Vec3 v1, Vec3 v2 ) { return v1.points[X]* v2.points[X] + v1.points[Y] * v2.points[Y] + v1.points[Z] * v2.points[Z]; }

Vec3 SUB( Vec3 v1, Vec3 v2 ) 
{ 
	Vec3 v;
	v.points[X] = v1.points[X] - v2.points[X];
	v.points[Y] = v1.points[Y] - v2.points[Y];
	v.points[Z] = v1.points[Z] - v2.points[Z]; 
	return v;
}

Vec3 ADD(Vec3 v1, Vec3 v2 ) 
{ 
	Vec3 v;
	v.points[X] = v1.points[X] + v2.points[X];
	v.points[Y] = v1.points[Y] + v2.points[Y];
	v.points[Z] = v1.points[Z] + v2.points[Z]; 
	return v;
}


Vec3 CROSS(Vec3 a, Vec3 b ) 
{ 
	Vec3 v;
	v.points[X] = a.points[Y] * b.points[Z] - a.points[Z] * b.points[Y];
	v.points[Y] = a.points[Z] * b.points[X] - a.points[X] * b.points[Z];
	v.points[Z] = a.points[X] * b.points[Y] - a.points[Y] * b.points[X]; 
	return v;
}

Vec3 MUL(Vec3 v,float f){ 
	v.points[X] = v.points[X] * f;
	v.points[Y] = v.points[Y] * f;
	v.points[Z] = v.points[Z] * f;
	return v;

}

inline int triangle_intersection( Vec3   V1,  // Triangle vertices
                           Vec3   V2,
                           Vec3   V3,
                           Vec3    O,  //Ray origin
                           Vec3    D,  //Ray direction
                           __global Vec3* int_point )
{
Vec3 e1, e2;  //Edge1, Edge2
Vec3 P, Q, T;
  float det, inv_det, u, v;
  float t;

  //Find vectors for two edges sharing V1
  e1 = SUB(V2, V1);
  e2 = SUB(V3, V1);
  //Begin calculating determinant - also used to calculate u parameter
  P = CROSS(D, e2);
  //if determinant is near zero, ray lies in plane of triangle
  det = DOT(e1, P);
  //NOT CULLING
  if(det > -EPSILON && det < EPSILON) return 0;
  inv_det = 1.f / det;

  //calculate distance from V1 to ray origin
  T = SUB(O, V1);

  //Calculate u parameter and test bound
  u = DOT(T, P) * inv_det;
  //The intersection lies outside of the triangle
  if(u < 0.f || u > 1.f) return 0;

  //Prepare to test v parameter
  Q = CROSS(T, e1);

  //Calculate V parameter and test bound
  v = DOT(D, Q) * inv_det;
  //The intersection lies outside of the triangle
  if(v < 0.f || u + v  > 1.f) return 0;

  t = DOT(e2, Q) * inv_det;

  if(t > EPSILON) { //ray intersection
    //*out = t;

	*int_point = ADD(O, MUL(D, t));

    return 1;
  }

  // No hit, no win
  return 0;
}


__kernel void RayTriangleIntersection(__global Triangle *input, 
									  __global Vec3* output_point,  
									  __global int* output_hit, 
									  int num_vertices, 
									  Vec3 ray_origin, 
									  Vec3 ray_direction)
{
#define RUN 128
	int k = get_global_id(0);

	if(k < num_vertices)
	{ 
		int j;
		int l = k * RUN;

		for(j = 0; j<RUN && (l+j)<num_vertices; ++j){
			output_hit[l + j] = triangle_intersection(input[l + j].vertex1, input[l + j].vertex2, input[l + j].vertex3, ray_origin, ray_direction, &output_point[l + j]);

		}
	}
}
