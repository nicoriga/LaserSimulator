#define EPSILON 0.000001

struct Vec3
{
	float x;	
	float y;	
	float z;	
};

struct Triangle
{ 
	Vec3 vertex1;
	Vec3 vertex2;
	Vec3 vertex3;
};

float DOT( Vec3 v1, Vec3 v2 ) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }

Vec3 SUB(Vec3 v1, Vec3 v2 ) 
{ 
	Vec3 v;
	v.x = v1.x - v2.x;
	v.y = v1.y - v2.y;
	v.z = v1.z - v2.z; 
	return v;
}

Vec3 ADD(Vec3 v1, Vec3 v2 ) 
{ 
	Vec3 v;
	v.x = v1.x + v2.x;
	v.y = v1.y + v2.y;
	v.z = v1.z + v2.z; 
	return v;
}


Vec3 CROSS(Vec3 a, Vec3 b ) 
{ 
	Vec3 v;
	v.x = a.y * b.z - a.z * b.y;
	v.y = a.z * b.x - a.x * b.z;
	v.z = a.x * b.y - a.y * b.x; 
	return v;
}

Vec3 MUL(Vec3 v,float f){ 
	v.x = v.x * f;
	v.y = v.y * f;
	v.z = v.z * f;
	return v;

}

int triangle_intersection( const Vec3   V1,  // Triangle vertices
                           const Vec3   V2,
                           const Vec3   V3,
                           const Vec3    O,  //Ray origin
                           const Vec3    D,  //Ray direction
                                 Vec3* int_point )
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

	int_point = &ADD(O, MUL(D, t));

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
	int k = get_global_id(0);

	if(k < num_vertices)
	{ 
		Triangle* triangle = &input[k];
		Vec3* intersection_point = &output_point[k];

		output_hit[k] = triangle_intersection(triangle->vertex1, triangle->vertex2, triangle->vertex3, ray_origin, ray_direction, intersection_point);

	}
}
