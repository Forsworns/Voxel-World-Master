#include "includes/CompFab.h"
#include "math.h"
#include "curand.h"
#include "curand_kernel.h"
#include "includes/cuda_math.h"

#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"
#include <vector>

#define RANDOM_SEEDS 1000
#define EPSILONF 0.000001
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

// check cuda calls for errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// generates a random float between 0 and 1
__device__ float generate( curandState* globalState , int ind) 
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState; 
    return RANDOM;
}
// set up random seed buffer
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 


__device__ bool inside(unsigned int numIntersections, bool double_thick) {
	// if (double_thick && numIntersections % 2 == 0) return (numIntersections / 2) % 2 == 1;
	if (double_thick) return (numIntersections / 2) % 2 == 1;
	return numIntersections % 2 == 1;
}

// adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool intersects(CompFab::Triangle &triangle, float3 dir, float3 pos) {
	float3 V1 = {triangle.m_v1.m_x, triangle.m_v1.m_y, triangle.m_v1.m_z};
	float3 V2 = {triangle.m_v2.m_x, triangle.m_v2.m_y, triangle.m_v2.m_z};
	float3 V3 = {triangle.m_v3.m_x, triangle.m_v3.m_y, triangle.m_v3.m_z};

	//Find vectors for two edges sharing V1
	float3 e1 = V2 - V1;
	float3 e2 = V3 - V1;
	
	// //Begin calculating determinant - also used to calculate u parameter 
	float3 P = cross(dir, e2);

	//if determinant is near zero, ray lies in plane of triangle 
	float det = dot(e1, P);
	
	//NOT CULLING
	if(det > -EPSILONF && det < EPSILONF) return false;
	float inv_det = 1.f / det;

	// calculate distance from V1 to ray origin
	float3 T = pos - V1;
	//Calculate u parameter and test bound
	float u = dot(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f) return false;

	//Prepare to test v parameter
	float3 Q = cross(T, e1);
	//Calculate V parameter and test bound
	float v = dot(dir, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f) return false;

	float t = dot(e2, Q) * inv_det;

	if(t > EPSILONF) { // ray intersection
		return true;
	}

	// No hit, no win
	return false;
}

// Decides whether or not each voxel is within the given mesh
__global__ void voxelize_kernel( 
	bool* R, CompFab::Triangle* triangles, const int numTriangles, 
	const float spacing, const float3 bottom_left,
	const int w, const int h, const int d, bool double_thick)
{
	// find the position of the voxel
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	// pick an arbitrary sampling direction
	float3 dir = make_float3(1.0, 0.0, 0.0);

	if ( (xIndex < w) && (yIndex < h) && (zIndex < d) )
	{
		// find linearlized index in final boolean array 
		unsigned int index_out = zIndex*(w*h)+yIndex*w + xIndex;
		
		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing*xIndex,bottom_left.y + spacing*yIndex,bottom_left.z + spacing*zIndex);

		// check if the voxel is inside of the mesh. 
		// if it is inside, then there should be an odd number of 
		// intersections with the surrounding mesh
		unsigned int intersections = 0;
		for (int i = 0; i < numTriangles; ++i)
			if (intersects(triangles[i], dir, pos))
				intersections += 1;

		// store answer
		R[index_out] = inside(intersections, double_thick);
	}
}


// Decides whether or not each voxel is within the given partially un-closed mesh
// checks a variety of directions and picks most common belief
__global__ void voxelize_kernel_open_mesh( 
	// triangles of the mesh being voxelized
	bool* R, CompFab::Triangle* triangles, const int numTriangles, 
	// information about how large the samples are and where they begin
	const float spacing, const float3 bottom_left,
	// number of voxels
	const int w, const int h, const int d, 
	// sampling information for multiple intersection rays
	const int samples, curandState* globalState, bool double_thick
	)
{
	// find the position of the voxel
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;
	
	if ( (xIndex < w) && (yIndex < h) && (zIndex < d) )
	{
		// find linearlized index in final boolean array
		unsigned int index_out = zIndex*(w*h)+yIndex*w + xIndex;
		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing*xIndex,bottom_left.y + spacing*yIndex,bottom_left.z + spacing*zIndex);
		float3 dir;

		// we will randomly sample 3D space by sending rays in randomized directions
		int votes = 0;
		float theta;
		float z;

		for (int j = 0; j < samples; ++j)
		{
			// compute the random direction. Convert from polar to euclidean to get an even distribution
			theta = generate(globalState, index_out % RANDOM_SEEDS) * 2.f * E_PI;
			z = generate(globalState, index_out % RANDOM_SEEDS) * 2.f - 1.f;

			dir.x = sqrt(1-z*z) * cosf(theta);
			dir.y = sqrt(1-z*z) * sinf(theta);
			dir.z = sqrt(1-z*z) * cosf(theta);

			// check if the voxel is inside of the mesh. 
			// if it is inside, then there should be an odd number of 
			// intersections with the surrounding mesh
			unsigned int intersections = 0;
			for (int i = 0; i < numTriangles; ++i)
				if (intersects(triangles[i], dir, pos)) 
					intersections += 1;
			if (inside(intersections, double_thick)) votes += 1;
		}
		// choose the most popular answer from all of the randomized samples
		R[index_out] = votes >= (samples / 2.f);
	}
}

// voxelize the given mesh with the given resolution and dimensions
void kernel_wrapper(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick)
{
	int blocksInX = (w+8-1)/8;
	int blocksInY = (h+8-1)/8;
	int blocksInZ = (d+8-1)/8;

	dim3 Dg(blocksInX, blocksInY, blocksInZ); // ��ò�Ҫ����256*�ֱ��ʣ�Զ���ڴ˻ᳬ��һ��grid��yz����������߳���Ŀ�������256*�ֱ����Ѿ����Դﵽ�����·��㷨800*��Ч�������Ǹ������ҵõ�����ʵ�ĵ�ģ��
	dim3 Db(8, 8, 8);

	curandState* devStates;

	// samples���������ѡȡ
	if (samples > 0) {
		// set up random numbers
		dim3 tpb(RANDOM_SEEDS,1,1);
	    cudaMalloc ( &devStates, RANDOM_SEEDS*sizeof( curandState ) );
	    // setup seeds
	    setup_kernel <<< 1, tpb >>> ( devStates, time(NULL) );
	}
	
	// set up boolean array on the GPU
	bool *gpu_inside_array;
	gpuErrchk( cudaMalloc( (void **)&gpu_inside_array, sizeof(bool) * w * h * d ) );
	gpuErrchk( cudaMemcpy( gpu_inside_array, g_voxelGrid->m_insideArray, sizeof(bool) * w * h * d, cudaMemcpyHostToDevice ) );

	// set up triangle array on the GPU
	CompFab::Triangle* triangle_array = &triangles[0];
	CompFab::Triangle* gpu_triangle_array;
	gpuErrchk( cudaMalloc( (void **)&gpu_triangle_array, sizeof(CompFab::Triangle) * triangles.size() ) );
	gpuErrchk( cudaMemcpy( gpu_triangle_array, triangle_array, sizeof(CompFab::Triangle) * triangles.size(), cudaMemcpyHostToDevice ) );

	// ���½�
	float3 lower_left = make_float3(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z);
		
	if (samples > 0) {
		voxelize_kernel_open_mesh<<< Dg, Db >>>(gpu_inside_array, gpu_triangle_array, triangles.size(), (float) g_voxelGrid->m_spacing, lower_left, w, h, d, samples, devStates, double_thick);
	} else {
		voxelize_kernel<<< Dg, Db >>>(gpu_inside_array, gpu_triangle_array, triangles.size(), (float) g_voxelGrid->m_spacing, lower_left, w, h, d, double_thick);
	}

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	gpuErrchk( cudaMemcpy( g_voxelGrid->m_insideArray, gpu_inside_array, sizeof(bool) * w * h * d, cudaMemcpyDeviceToHost ) );

	gpuErrchk( cudaFree(gpu_inside_array) );
	gpuErrchk( cudaFree(gpu_triangle_array) );
}
//----------------------------------------------------------------------------------------------------------------------------------
#define VOXELIZER_HASH_TABLE_SIZE (32768)
#define VOXELIZER_EPSILON (0.0000001)
namespace Device {
	struct Triangle
	{
		CompFab::Vec3 m_v1; 
		CompFab::Vec3 m_v2;
		CompFab::Vec3 m_v3;
	};
};


// ����������Χ�У�aabb�У��������ǰ�Χ�жԽ����ϵ�������
typedef struct vx_aabb {
	float3 min;
	float3 max;
} vx_aabb_t;

#define FIGURE_OUT_DIRECTION(n,mi,ma,h)	\
	if (n > 0.0f){mi = -h;ma = h;}		\
	else {mi = h;ma = -h;}

// �ж�ƽ���Ƿ����Χ���ཻ
__device__ int vx__plane_box_overlap(float3 normal, float d, float3 halfboxsize)
{
	float3 vmin, vmax;												// ��Ϊƽ�汻�ƶ�����ԭ�㣬��Χ������Ҳ����ԭ�㴦������Ҫ�Ƚϡ�halfsize																						

	FIGURE_OUT_DIRECTION(normal.x, vmin.x, vmax.x, halfboxsize.x);	// ���������η������ķ���ȷ���˰�Χ��Ӧ��ѡ��������С�ĵ�
	FIGURE_OUT_DIRECTION(normal.y, vmin.y, vmax.y, halfboxsize.y);
	FIGURE_OUT_DIRECTION(normal.z, vmin.z, vmax.z, halfboxsize.z);


	if (dot(normal, vmin) + d > 0.0f) {								// ���㵽ƽ������뵽�������Ĵ�С��ϵȷ���Ƿ��ཻ
		return false;
	}

	if (dot(normal, vmax) + d >= 0.0f) {
		return true;
	}

	return false;
}

#define VX_FINDMINMAX(x0, x1, x2, min, max) \
    min = max = x0;                         \
    if (x1 < min) min = x1;                 \
    if (x1 > max) max = x1;                 \
    if (x2 < min) min = x2;                 \
    if (x2 > max) max = x2;
// �ж�������Ƭ���Χ���Ƿ��ཻ
__device__ int vx__triangle_box_overlap(float3 boxcenter,
	float3 halfboxsize,
	Device::Triangle* triangle)
{
	float3 v1, v2, v3, normal, e1, e2, e3;
	float min, max, d, p1, p2, p3, rad, fex, fey, fez;

	v1 = { triangle->m_v1.m_x,triangle->m_v1.m_y,triangle->m_v1.m_z };
	v2 = { triangle->m_v2.m_x,triangle->m_v2.m_y,triangle->m_v2.m_z };
	v3 = { triangle->m_v3.m_x,triangle->m_v3.m_y,triangle->m_v3.m_z };

	v1 -= boxcenter;															// ���������ƶ���ԭ��
	v2 -= boxcenter;
	v3 -= boxcenter;

	e1 = v2 - v1;																	// ���������θ���
	e2 = v3 - v2;
	e3 = v1 - v3;

	VX_FINDMINMAX(v1.x, v2.x, v3.x, min, max);									// �����������Ϊ��������ȫ���������һ��
	if (min > halfboxsize.x || max < -halfboxsize.x) {
		return false;
	}

	VX_FINDMINMAX(v1.y, v2.y, v3.y, min, max);
	if (min > halfboxsize.y || max < -halfboxsize.y) {
		return false;
	}

	VX_FINDMINMAX(v1.z, v2.z, v3.z, min, max);
	if (min > halfboxsize.z || max < -halfboxsize.z) {
		return false;
	}

	normal = cross(e1, e2);														// �������淨����
	d = -dot(normal, v1);

	if (!vx__plane_box_overlap(normal, d, halfboxsize)) {						// ���ݾ����ж������룬���������δ���������
		return false;
	}

	return true;																
}

// �������������
__device__ float vx__triangle_area(Device::Triangle* triangle) {
	float3 v1 = { triangle->m_v1.m_x,triangle->m_v1.m_y,triangle->m_v1.m_z };
	float3 v2 = { triangle->m_v2.m_x,triangle->m_v2.m_y,triangle->m_v2.m_z };
	float3 v3 = { triangle->m_v3.m_x,triangle->m_v3.m_y,triangle->m_v3.m_z };

	float3 ab = v2 - v1;														// �����ab��ac
	float3 ac = v3 - v1;

	float a0 = ab.y * ac.z - ab.z * ac.y;										// ab,ac���
	float a1 = ab.z * ac.x - ab.x * ac.z;
	float a2 = ab.x * ac.y - ab.y * ac.x;

	return sqrtf(powf(a0, 2.f) + powf(a1, 2.f) + powf(a2, 2.f)) * 0.5f;			// ���ò��(|ab|*|ac|*sinA)/2�������
}

// ��ʼ��aabb��Χ�У���֮��������޵㣬�ᵼ��max<min
__device__ void vx__aabb_init(vx_aabb_t* aabb)
{
	aabb->max.x = aabb->max.y = aabb->max.z = -INFINITY;
	aabb->min.x = aabb->min.y = aabb->min.z = INFINITY;
}


#define VX_MIN(a, b) (a > b ? b : a)
#define VX_MAX(a, b) (a > b ? a : b)
__device__ vx_aabb_t vx__triangle_aabb(Device::Triangle* triangle)
{
	vx_aabb_t aabb;

	vx__aabb_init(&aabb);

	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v1.m_x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v1.m_x);
	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v2.m_x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v2.m_x);
	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v3.m_x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v3.m_x);

	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v1.m_y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v1.m_y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v2.m_y);
	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v2.m_y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v3.m_y);
	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v3.m_y);

	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v1.m_z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v1.m_z);
	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v2.m_z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v2.m_z);
	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v3.m_z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v3.m_z);

	return aabb;
}

// ȷ����Χ������
__device__ float3 vx__aabb_center(vx_aabb_t* a)
{
	float3 boxcenter = 0.5f * (a->min + a->max);
	return boxcenter;
}

// ȡ��Χ�д�С��һ��
__device__ float3 vx__aabb_half_size(vx_aabb_t* a)
{
	float3 size;

	size.x = fabs(a->max.x - a->min.x) * 0.5f;
	size.y = fabs(a->max.y - a->min.y) * 0.5f;
	size.z = fabs(a->max.z - a->min.z) * 0.5f;

	return size;
}

// ��min�������ϻ�����ȡ������������������0���Ƿ����;��������Ƿ���Ч
__device__ float vx__map_to_voxel(float position, float voxelSize, bool min)
{	// ���÷��ź����������ػ����������������0.5���ϣ�����ȷ��֮��ʹ��ʱ����ʹ��������ض�������һ��ƽ���ϣ��γ����ٺ��Ϊ1�����ز�										
	float vox = (position + (position < 0.f ? -1.f : 1.f) * voxelSize * 0.5f) / voxelSize;
	// float vox = position / voxelSize; // �����ֶ�������������
	return (min ? floor(vox) : ceil(vox)) * voxelSize;
}

// ���ػ��ĺ��Ĵ���
__global__ void vx__voxelize(int *times, bool* R, float3 vs, float3 hvs, float precision, CompFab::Triangle* triangles, int triangles_num, int w, int h, int d, float spacing, float3 lower_left)
{
	//printf("the thread blockIdx.x %d, blockIdx.y %d blockIdx.z %d\n", blockIdx.x,blockIdx.y, blockIdx.z);

	unsigned int insideIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	unsigned int outsideIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
	unsigned int blockSize = blockDim.x*blockDim.y*blockDim.z;
	unsigned int i = insideIndex + outsideIndex*blockSize;

	if (i < triangles_num) {
		CompFab::Vec3 p1 = triangles[i].m_v1;							// ���������������Ƭ
		CompFab::Vec3 p2 = triangles[i].m_v2;
		CompFab::Vec3 p3 = triangles[i].m_v3;
		Device::Triangle triangle = { p1, p2, p3 };
		if (vx__triangle_area(&triangle) < VOXELIZER_EPSILON) {			// ���������С��������
			return;
		}
		vx_aabb_t aabb = vx__triangle_aabb(&triangle);					// �����ε�aabb��Χ��

		//aabb.min.x = vx__map_to_voxel(aabb.min.x, vs.x, true);		// ���������λ�õ���������ֹ�����������
		//aabb.min.y = vx__map_to_voxel(aabb.min.y, vs.y, true);
		//aabb.min.z = vx__map_to_voxel(aabb.min.z, vs.z, true);

		//aabb.max.x = vx__map_to_voxel(aabb.max.x, vs.x, false);
		//aabb.max.y = vx__map_to_voxel(aabb.max.y, vs.y, false);
		//aabb.max.z = vx__map_to_voxel(aabb.max.z, vs.z, false);		// ����ʹ����map_to_voxel��һ������falseһ������true�����ٻ��γ�һ������

		int temp = 0,temp2=0;
		// �������Χ�����ཻ�ĸ��
		for (float x = aabb.min.x; x <= aabb.max.x; x += vs.x) {
			for (float y = aabb.min.y; y <= aabb.max.y; y += vs.y) {
				for (float z = aabb.min.z; z <= aabb.max.z; z += vs.z) {
					vx_aabb_t saabb;
					temp++;
					saabb.min.x = x - hvs.x;
					saabb.min.y = y - hvs.y;
					saabb.min.z = z - hvs.z;
					saabb.max.x = x + hvs.x;
					saabb.max.y = y + hvs.y;
					saabb.max.z = z + hvs.z;

					float3 boxcenter = vx__aabb_center(&saabb);			// ��aabb�ĺô���������Ҳ���Կ�����aabb���ӣ�
					float3 halfsize = vx__aabb_half_size(&saabb);

					// HACK: some holes might appear, this
					// precision factor reduces the artifact
					halfsize.x += precision;
					halfsize.y += precision;
					halfsize.z += precision;
					if (vx__triangle_box_overlap(boxcenter, halfsize, &triangle)) {
						temp2++;
						int3 pos;
						pos.x = floor((boxcenter.x - lower_left.x) / spacing) + 1;
						pos.y = floor((boxcenter.y -lower_left.y) / spacing) + 1;
						pos.z = floor((boxcenter.z - lower_left.z) / spacing) + 1;
						if (pos.x < w && pos.y < h && pos.z < d) {
							int index = pos.z*w*h + pos.y*w + pos.x;
							atomicAdd(times, 1);
							R[index] = true; 
							//if(R[index]) atomicAdd(times, 1);
						}
					}
					
				}
			}
		}
		// printf("thread %d with %d of %d\n", i, temp2, temp);
	}
}

void kernel_wrapper_2(int w, int h, int d, float precision, CompFab::VoxelGrid* g_voxelGrid, std::vector<CompFab::Triangle> triangles) {
	//  get the voxel result
	std::cout << "����Ϊ" << g_voxelGrid->m_spacing<<std::endl;
	int* times = new int(0);
	int* gpu_times;
	gpuErrchk(cudaMalloc((void **)&gpu_times, sizeof(int) ));
	gpuErrchk(cudaMemcpy(gpu_times, times, sizeof(int), cudaMemcpyHostToDevice));


	float3 vs = { g_voxelGrid->m_spacing, g_voxelGrid->m_spacing, g_voxelGrid->m_spacing };
	float3 hvs = 0.5*vs;

	int blockSize = ceil(pow(triangles.size() / 512.0, 1.0 / 3));
	dim3 Dg(blockSize, blockSize, blockSize);
	dim3 Db(8, 8, 8);
	
	CompFab::Triangle* triangle_array = &triangles[0];
	CompFab::Triangle* gpu_triangle_array;
	gpuErrchk(cudaMalloc((void **)&gpu_triangle_array, sizeof(CompFab::Triangle) * triangles.size()));
	gpuErrchk(cudaMemcpy(gpu_triangle_array, triangle_array, sizeof(CompFab::Triangle) * triangles.size(), cudaMemcpyHostToDevice));

	// set up boolean array on the GPU��
	bool *gpu_inside_array;
	gpuErrchk(cudaMalloc((void **)&gpu_inside_array, sizeof(bool) * w * h * d));
	float3 lower_left = make_float3(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z);
	vx__voxelize <<< Dg, Db >>> (gpu_times, gpu_inside_array, vs, hvs, precision, gpu_triangle_array, triangles.size(), w, h, d, g_voxelGrid->m_spacing, lower_left);
	
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(g_voxelGrid->m_insideArray, gpu_inside_array, sizeof(bool) * w * h * d, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(times, gpu_times, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout <<w*h*d <<"times " << *times <<" lx "<< lower_left.x << " ly "<<lower_left.y<<" lz "<<lower_left.z <<" "<< lower_left.x + 16* g_voxelGrid->m_spacing << std::endl;
	gpuErrchk(cudaFree(gpu_times));

	int j = 0;
	for (int i = 0; i < w*h*d; ++i) {
		if (g_voxelGrid->m_insideArray[i]){
			++j;
		}
	}
	std::cout << "the voxel is " << j << std::endl;

	gpuErrchk(cudaFree(gpu_inside_array));
	gpuErrchk(cudaFree(gpu_triangle_array));//I called it in my real codes, I forgot add it. Modified
}
