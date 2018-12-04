#include "includes/CompFab.h"
#include "includes/marchingTable.h"
#include "math.h"
#include "curand.h"
#include "curand_kernel.h"
#include "includes/cuda_math.h"

#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"
#include <vector>

namespace Device
{
struct Triangle
{
	float3 m_v1;
	float3 m_v2;
	float3 m_v3;
	__device__ Triangle() {}
	__device__ Triangle(float3 v1, float3 v2, float3 v3) : m_v1(v1), m_v2(v2), m_v3(v3) {}
};
struct Point
{
	float3 pos;
	float3 normal;
	double weight;
	__device__ Point() : pos(make_float3(0, 0, 0)), normal(make_float3(0, 0, 0)), weight(0) {}
	__device__ Point(float3 p, float3 n, double w) : pos(p), normal(n), weight(w) {}
};
struct TrianglePN
{
	Triangle pos;
	Triangle normal;
	TrianglePN() {}
	__device__ TrianglePN(Triangle p, Triangle n) : pos(p), normal(n) {}
};
}; // namespace Device

#define RANDOM_SEEDS 1000
#define EPSILONF 0.000001
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

// check cuda calls for errors
#define gpuErrchk(ans)                        \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

// generates a random float between 0 and 1
__device__ float generate(curandState *globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}
// set up random seed buffer
__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ bool inside(unsigned int numIntersections, bool double_thick)
{
	// if (double_thick && numIntersections % 2 == 0) return (numIntersections / 2) % 2 == 1;
	if (double_thick)
		return (numIntersections / 2) % 2 == 1;
	return numIntersections % 2 == 1;
}

// adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool intersects(CompFab::Triangle &triangle, float3 dir, float3 pos)
{
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
	if (det > -EPSILONF && det < EPSILONF)
		return false;
	float inv_det = 1.f / det;

	// calculate distance from V1 to ray origin
	float3 T = pos - V1;
	//Calculate u parameter and test bound
	float u = dot(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if (u < 0.f || u > 1.f)
		return false;

	//Prepare to test v parameter
	float3 Q = cross(T, e1);
	//Calculate V parameter and test bound
	float v = dot(dir, Q) * inv_det;
	//The intersection lies outside of the triangle
	if (v < 0.f || u + v > 1.f)
		return false;

	float t = dot(e2, Q) * inv_det;

	if (t > EPSILONF)
	{ // ray intersection
		return true;
	}

	// No hit, no win
	return false;
}

// Decides whether or not each voxel is within the given mesh
__global__ void voxelize_kernel(
	bool *R, CompFab::Triangle *triangles, const int numTriangles,
	const float spacing, const float3 bottom_left,
	const int w, const int h, const int d, bool double_thick)
{
	// find the position of the voxel
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	// pick an arbitrary sampling direction
	float3 dir = make_float3(1.0, 0.0, 0.0);

	if ((xIndex < w) && (yIndex < h) && (zIndex < d))
	{
		// find linearlized index in final boolean array
		unsigned int index_out = zIndex * (w * h) + yIndex * w + xIndex;

		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing * xIndex, bottom_left.y + spacing * yIndex, bottom_left.z + spacing * zIndex);

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
	bool *R, CompFab::Triangle *triangles, const int numTriangles,
	// information about how large the samples are and where they begin
	const float spacing, const float3 bottom_left,
	// number of voxels
	const int w, const int h, const int d,
	// sampling information for multiple intersection rays
	const int samples, curandState *globalState, bool double_thick)
{
	// find the position of the voxel
	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int zIndex = blockDim.z * blockIdx.z + threadIdx.z;

	if ((xIndex < w) && (yIndex < h) && (zIndex < d))
	{
		// find linearlized index in final boolean array
		unsigned int index_out = zIndex * (w * h) + yIndex * w + xIndex;
		// find world space position of the voxel
		float3 pos = make_float3(bottom_left.x + spacing * xIndex, bottom_left.y + spacing * yIndex, bottom_left.z + spacing * zIndex);
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

			dir.x = sqrt(1 - z * z) * cosf(theta);
			dir.y = sqrt(1 - z * z) * sinf(theta);
			dir.z = sqrt(1 - z * z) * cosf(theta);

			// check if the voxel is inside of the mesh.
			// if it is inside, then there should be an odd number of
			// intersections with the surrounding mesh
			unsigned int intersections = 0;
			for (int i = 0; i < numTriangles; ++i)
				if (intersects(triangles[i], dir, pos))
					intersections += 1;
			if (inside(intersections, double_thick))
				votes += 1;
		}
		// choose the most popular answer from all of the randomized samples
		R[index_out] = votes >= (samples / 2.f);
	}
}

// voxelize the given mesh with the given resolution and dimensions
void kernel_wrapper_1(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick)
{
	int blocksInX = (w + 8 - 1) / 8;
	int blocksInY = (h + 8 - 1) / 8;
	int blocksInZ = (d + 8 - 1) / 8;

	dim3 Dg(blocksInX, blocksInY, blocksInZ); // 最好不要超过256*分辨率，远高于此会超出一个grid的yz分量的最大线程数目，这个的256*分辨率已经可以达到媲美下方算法800*的效果，但是更慢而且得到的是实心的模型
	dim3 Db(8, 8, 8);

	curandState *devStates;

	// samples条光线随机选取
	if (samples > 0)
	{
		// set up random numbers
		dim3 tpb(RANDOM_SEEDS, 1, 1);
		cudaMalloc(&devStates, RANDOM_SEEDS * sizeof(curandState));
		// setup seeds
		setup_kernel<<<1, tpb>>>(devStates, time(NULL));
	}

	// set up boolean array on the GPU
	bool *gpu_inside_array;
	gpuErrchk(cudaMalloc((void **)&gpu_inside_array, sizeof(bool) * w * h * d));
	gpuErrchk(cudaMemcpy(gpu_inside_array, g_voxelGrid->m_insideArray, sizeof(bool) * w * h * d, cudaMemcpyHostToDevice));

	// set up triangle array on the GPU
	CompFab::Triangle *triangle_array = &triangles[0];
	CompFab::Triangle *gpu_triangle_array;
	gpuErrchk(cudaMalloc((void **)&gpu_triangle_array, sizeof(CompFab::Triangle) * triangles.size()));
	gpuErrchk(cudaMemcpy(gpu_triangle_array, triangle_array, sizeof(CompFab::Triangle) * triangles.size(), cudaMemcpyHostToDevice));

	// 左下角
	float3 lower_left = make_float3(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z);

	if (samples > 0)
	{
		voxelize_kernel_open_mesh<<<Dg, Db>>>(gpu_inside_array, gpu_triangle_array, triangles.size(), (float)g_voxelGrid->m_spacing, lower_left, w, h, d, samples, devStates, double_thick);
	}
	else
	{
		voxelize_kernel<<<Dg, Db>>>(gpu_inside_array, gpu_triangle_array, triangles.size(), (float)g_voxelGrid->m_spacing, lower_left, w, h, d, double_thick);
	}

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(g_voxelGrid->m_insideArray, gpu_inside_array, sizeof(bool) * w * h * d, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(gpu_inside_array));
	gpuErrchk(cudaFree(gpu_triangle_array));
}

//----------------------------------------------------------------------------------------------------------------------------------

#define VOXELIZER_HASH_TABLE_SIZE (32768)
#define VOXELIZER_EPSILON (0.0000001)

// 用于轴对齐包围盒（aabb盒），这里是包围盒对角线上的两个点
typedef struct vx_aabb
{
	float3 min;
	float3 max;
} vx_aabb_t;

#define FIGURE_OUT_DIRECTION(n, mi, ma, h) \
	if (n > 0.0f)                          \
	{                                      \
		mi = -h;                           \
		ma = h;                            \
	}                                      \
	else                                   \
	{                                      \
		mi = h;                            \
		ma = -h;                           \
	}

// 判断平面是否与包围盒相交
__device__ int vx__plane_box_overlap(float3 normal, float d, float3 halfboxsize)
{
	float3 vmin, vmax; // 因为平面被移动到了原点，包围盒中心也放在原点处理所以要比较±halfsize

	FIGURE_OUT_DIRECTION(normal.x, vmin.x, vmax.x, halfboxsize.x); // 根据三角形法向量的方向确定了包围盒应该选择的最大最小的点
	FIGURE_OUT_DIRECTION(normal.y, vmin.y, vmax.y, halfboxsize.y);
	FIGURE_OUT_DIRECTION(normal.z, vmin.z, vmax.z, halfboxsize.z);

	if (dot(normal, vmin) + d > 0.0f)
	{ // 计算到平面距离与到顶点距离的大小关系确定是否相交
		return false;
	}

	if (dot(normal, vmax) + d >= 0.0f)
	{
		return true;
	}

	return false;
}

#define VX_FINDMINMAX(x0, x1, x2, min, max) \
	min = max = x0;                         \
	if (x1 < min)                           \
		min = x1;                           \
	if (x1 > max)                           \
		max = x1;                           \
	if (x2 < min)                           \
		min = x2;                           \
	if (x2 > max)                           \
		max = x2;
// 判断三角面片与包围盒是否相交
__device__ int vx__triangle_box_overlap(float3 boxcenter,
										float3 halfboxsize,
										Device::Triangle *triangle)
{
	float3 v1, v2, v3, normal, e1, e2, e3;
	float min, max, d, p1, p2, p3, rad, fex, fey, fez;

	v1 = triangle->m_v1;
	v2 = triangle->m_v2;
	v3 = triangle->m_v3;

	v1 -= boxcenter; // 将三角形移动到原点
	v2 -= boxcenter;
	v3 -= boxcenter;

	e1 = v2 - v1; // 计算三角形各边
	e2 = v3 - v2;
	e3 = v1 - v3;

	VX_FINDMINMAX(v1.x, v2.x, v3.x, min, max); // 以下三种情况为三角形完全在正方体的一侧
	if (min > halfboxsize.x || max < -halfboxsize.x)
	{
		return false;
	}

	VX_FINDMINMAX(v1.y, v2.y, v3.y, min, max);
	if (min > halfboxsize.y || max < -halfboxsize.y)
	{
		return false;
	}

	VX_FINDMINMAX(v1.z, v2.z, v3.z, min, max);
	if (min > halfboxsize.z || max < -halfboxsize.z)
	{
		return false;
	}

	normal = cross(e1, e2); // 三角形面法向量
	d = -dot(normal, v1);

	if (!vx__plane_box_overlap(normal, d, halfboxsize))
	{ // 根据距离判断是相离，还是三角形穿过正方体
		return false;
	}

	return true;
}

// 计算三角形面积
__device__ float vx__triangle_area(Device::Triangle *triangle)
{
	float3 v1 = triangle->m_v1;
	float3 v2 = triangle->m_v2;
	float3 v3 = triangle->m_v3;

	float3 ab = v2 - v1; // 计算边ab，ac
	float3 ac = v3 - v1;

	float a0 = ab.y * ac.z - ab.z * ac.y; // ab,ac叉积
	float a1 = ab.z * ac.x - ab.x * ac.z;
	float a2 = ab.x * ac.y - ab.y * ac.x;

	return sqrtf(powf(a0, 2.f) + powf(a1, 2.f) + powf(a2, 2.f)) * 0.5f; // 利用叉积(|ab|*|ac|*sinA)/2计算面积
}

// 初始化aabb包围盒，若之后盒子中无点，会导致max<min
__device__ void vx__aabb_init(vx_aabb_t *aabb)
{
	aabb->max.x = aabb->max.y = aabb->max.z = -INFINITY;
	aabb->min.x = aabb->min.y = aabb->min.z = INFINITY;
}

#define VX_MIN(a, b) (a > b ? b : a)
#define VX_MAX(a, b) (a > b ? a : b)
__device__ vx_aabb_t vx__triangle_aabb(Device::Triangle *triangle)
{
	vx_aabb_t aabb;

	vx__aabb_init(&aabb);

	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v1.x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v1.x);
	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v2.x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v2.x);
	aabb.max.x = VX_MAX(aabb.max.x, triangle->m_v3.x);
	aabb.min.x = VX_MIN(aabb.min.x, triangle->m_v3.x);

	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v1.y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v1.y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v2.y);
	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v2.y);
	aabb.max.y = VX_MAX(aabb.max.y, triangle->m_v3.y);
	aabb.min.y = VX_MIN(aabb.min.y, triangle->m_v3.y);

	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v1.z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v1.z);
	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v2.z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v2.z);
	aabb.max.z = VX_MAX(aabb.max.z, triangle->m_v3.z);
	aabb.min.z = VX_MIN(aabb.min.z, triangle->m_v3.z);

	return aabb;
}

// 确定包围盒中心
__device__ float3 vx__aabb_center(vx_aabb_t *a)
{
	float3 boxcenter = 0.5f * (a->min + a->max);
	return boxcenter;
}

// 取包围盒大小的一半
__device__ float3 vx__aabb_half_size(vx_aabb_t *a)
{
	float3 size;

	size.x = fabs(a->max.x - a->min.x) * 0.5f;
	size.y = fabs(a->max.y - a->min.y) * 0.5f;
	size.z = fabs(a->max.z - a->min.z) * 0.5f;

	return size;
}

// 用min决定向上或向下取整，可以用来决定对0处是否填充和决定提升是否有效
__device__ float vx__map_to_voxel(float position, float voxelSize, bool min)
{ // 利用符号函数，将体素化后的坐标提升到±0.5以上，这样确保之后使用时可以使得两个点必定不会在一个平面上，形成至少厚度为1的体素层
	float vox = (position + (position < 0.f ? -1.f : 1.f) * voxelSize * 0.5f) / voxelSize;
	// float vox = position / voxelSize; // 何种手段提升体素坐标
	return (min ? floor(vox) : ceil(vox)) * voxelSize;
}

// 体素化的核心代码
__global__ void vx__voxelize(int *times, bool *R, float3 vs, float3 hvs, float precision, CompFab::Triangle *triangles, int triangles_num, int w, int h, int d, float spacing, float3 lower_left)
{
	//printf("the thread blockIdx.x %d, blockIdx.y %d blockIdx.z %d\n", blockIdx.x,blockIdx.y, blockIdx.z);

	unsigned int insideIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	unsigned int outsideIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
	unsigned int i = insideIndex + outsideIndex * blockSize;

	if (i < triangles_num)
	{
		float3 p1 = make_float3(triangles[i].m_v1.m_x, triangles[i].m_v1.m_y, triangles[i].m_v1.m_z); // 构造出了三角形面片
		float3 p2 = make_float3(triangles[i].m_v2.m_x, triangles[i].m_v2.m_y, triangles[i].m_v2.m_z);
		float3 p3 = make_float3(triangles[i].m_v3.m_x, triangles[i].m_v3.m_y, triangles[i].m_v3.m_z);
		Device::Triangle triangle = {p1, p2, p3};
		if (vx__triangle_area(&triangle) < VOXELIZER_EPSILON)
		{ // 跳过面积过小的三角形
			return;
		}
		vx_aabb_t aabb = vx__triangle_aabb(&triangle); // 三角形的aabb包围盒

		//aabb.min.x = vx__map_to_voxel(aabb.min.x, vs.x, true);		// 不做坐标的位置的扩增，防止出来的面过厚
		//aabb.min.y = vx__map_to_voxel(aabb.min.y, vs.y, true);
		//aabb.min.z = vx__map_to_voxel(aabb.min.z, vs.z, true);

		//aabb.max.x = vx__map_to_voxel(aabb.max.x, vs.x, false);
		//aabb.max.y = vx__map_to_voxel(aabb.max.y, vs.y, false);
		//aabb.max.z = vx__map_to_voxel(aabb.max.z, vs.z, false);		// 由于使用了map_to_voxel，一个传入false一个传入true，至少会形成一层体素

		int temp = 0, temp2 = 0;
		// 填补整个包围盒中相交的格点
		for (float x = aabb.min.x; x <= aabb.max.x; x += vs.x)
		{
			for (float y = aabb.min.y; y <= aabb.max.y; y += vs.y)
			{
				for (float z = aabb.min.z; z <= aabb.max.z; z += vs.z)
				{
					vx_aabb_t saabb;
					temp++;
					saabb.min.x = x - hvs.x;
					saabb.min.y = y - hvs.y;
					saabb.min.z = z - hvs.z;
					saabb.max.x = x + hvs.x;
					saabb.max.y = y + hvs.y;
					saabb.max.z = z + hvs.z;

					float3 boxcenter = vx__aabb_center(&saabb); // 用aabb的好处就是体素也可以看成是aabb盒子！
					float3 halfsize = vx__aabb_half_size(&saabb);

					// HACK: some holes might appear, this
					// precision factor reduces the artifact
					halfsize.x += precision;
					halfsize.y += precision;
					halfsize.z += precision;
					if (vx__triangle_box_overlap(boxcenter, halfsize, &triangle))
					{
						temp2++;
						int3 pos;
						pos.x = floor((boxcenter.x - lower_left.x) / spacing) + 1;
						pos.y = floor((boxcenter.y - lower_left.y) / spacing) + 1;
						pos.z = floor((boxcenter.z - lower_left.z) / spacing) + 1;
						if (pos.x < w && pos.y < h && pos.z < d)
						{
							int index = pos.z * w * h + pos.y * w + pos.x;
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

void kernel_wrapper_2(int w, int h, int d, float precision, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles)
{
	//  get the voxel result
	std::cout << "比例为" << g_voxelGrid->m_spacing << std::endl;
	int *times = new int(0);
	int *gpu_times;
	gpuErrchk(cudaMalloc((void **)&gpu_times, sizeof(int)));
	gpuErrchk(cudaMemcpy(gpu_times, times, sizeof(int), cudaMemcpyHostToDevice));

	float3 vs = {g_voxelGrid->m_spacing, g_voxelGrid->m_spacing, g_voxelGrid->m_spacing};
	float3 hvs = 0.5 * vs;

	int blockSize = ceil(pow(triangles.size() / 512.0, 1.0 / 3));
	dim3 Dg(blockSize, blockSize, blockSize);
	dim3 Db(8, 8, 8);

	CompFab::Triangle *triangle_array = &triangles[0];
	CompFab::Triangle *gpu_triangle_array;
	gpuErrchk(cudaMalloc((void **)&gpu_triangle_array, sizeof(CompFab::Triangle) * triangles.size()));
	gpuErrchk(cudaMemcpy(gpu_triangle_array, triangle_array, sizeof(CompFab::Triangle) * triangles.size(), cudaMemcpyHostToDevice));

	// set up boolean array on the GPU，
	bool *gpu_inside_array;
	gpuErrchk(cudaMalloc((void **)&gpu_inside_array, sizeof(bool) * w * h * d));
	float3 lower_left = make_float3(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z);
	vx__voxelize<<<Dg, Db>>>(gpu_times, gpu_inside_array, vs, hvs, precision, gpu_triangle_array, triangles.size(), w, h, d, g_voxelGrid->m_spacing, lower_left);

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(g_voxelGrid->m_insideArray, gpu_inside_array, sizeof(bool) * w * h * d, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaMemcpy(times, gpu_times, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << w * h * d << "times " << *times << " lx " << lower_left.x << " ly " << lower_left.y << " lz " << lower_left.z << " " << lower_left.x + 16 * g_voxelGrid->m_spacing << std::endl;
	gpuErrchk(cudaFree(gpu_times));

	int j = 0;
	for (int i = 0; i < w * h * d; ++i)
	{
		if (g_voxelGrid->m_insideArray[i])
		{
			++j;
		}
	}
	std::cout << "the voxel is " << j << std::endl;

	gpuErrchk(cudaFree(gpu_inside_array));
	gpuErrchk(cudaFree(gpu_triangle_array));
}

//---------------------------------------------------------------------------------------------------------------------------------
#define FOUT_LINE(a) fout << a << endl;
#define FOUT_POINT(a, b, c) fout << a << " " << b << " " << c << endl;
#define NONE_ZERO 0.0001f

#define INF 10000;

// 线性插值函数
__device__ Device::Point interpolate(const Device::Point &p1, const Device::Point &p2, const float &threhold)
{
	Device::Point point;
	if (abs(threhold - p1.weight) < NONE_ZERO)
	{
		return p1;
	}
	if (abs(threhold - p2.weight) < NONE_ZERO)
	{
		return p2;
	}
	if (abs(p2.weight - p1.weight) < NONE_ZERO)
	{
		return p1;
	}
	float weight = (threhold - p1.weight) / (p2.weight - p1.weight);

	// float weight = 0.5;
	point.pos = p1.pos + weight * (p2.pos - p1.pos);
	point.normal = p1.normal + weight * (p2.normal - p1.normal);
	normalize(point.normal);
	return point;
}

// 求解各体素格点的权值
__global__ void gridBuilder(int *count, Device::Point *grids, Device::Point *voxelPointMap, double scale, float3 orig, int3 dim, bool *voxel)
{
	int3 NEIGHBOR[8] = {{-1, -1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 0}, {-1, -1, -1}, {-1, 0, -1}, {0, -1, -1}, {0, 0, -1}};
	unsigned int insideIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	unsigned int outsideIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
	long long int index = insideIndex + outsideIndex * blockSize;
	int z = index / (dim.x * dim.y);
	int y = (index % (dim.x * dim.y)) / dim.x;
	int x = (index % (dim.x * dim.y)) % dim.x;
	int3 pos = make_int3(x, y, z);
	if (pos.x < dim.x && pos.y < dim.y && pos.z < dim.z)
	{
		grids[index].pos = make_float3(pos.x * scale + orig.x, pos.y * scale + orig.y, pos.z * scale + orig.z);
		int neighborNum = 0;
		float3 neighborPos = {0, 0, 0};
		float3 neighborNormal = {0, 0, 0};
		// 应该根据最近的几个点计算weight。这里改用最近一层的八个点：利用之前voxel的信息，直接找顶点附近的八个邻居，注意应该是顶点对应体素点和顶点想要的邻居体素点之间的关系
		for (int i = 0; i < 8; ++i)
		{
			int3 tempPos = pos + NEIGHBOR[i];
			if (tempPos.x < 0 || tempPos.y < 0 || tempPos.z < 0 ||
				tempPos.x >= dim.x || tempPos.y >= dim.y || tempPos.z >= dim.z ||
				!voxel[tempPos.z * dim.x * dim.y + tempPos.y * dim.x + tempPos.x])
			{
				continue;
			}
			else
			{
				Device::Point tempPoint = voxelPointMap[tempPos.z * dim.x * dim.y + tempPos.y * dim.x + tempPos.x];
				neighborPos = neighborPos + tempPoint.pos;
				neighborNormal = neighborNormal + tempPoint.normal;
				neighborNum++;
			}
		}
		// 如果是邻居在voxel时是没填满的部分，则跳过
		if (neighborNum == 0)
		{
			grids[index].weight = INF;
			return;
		}
		else
		{
			atomicAdd(count, 1);
			normalize(neighborNormal);
			neighborPos /= neighborNum;
			grids[index].weight = dot((grids[index].pos - neighborPos), neighborNormal);
		}
	}
}

#define INDEX_BUILD(a, b)                  \
	if (vertexOfGrid[a].weight < threhold) \
	{                                      \
		cubeIndex |= b;                    \
	}

#define VERTEX_LIST_BUILD(a, b, c, d)                                              \
	if (edgeTableGPU[cubeIndex] & b)                                               \
	{                                                                              \
		vertexOnEdge[a] = interpolate(vertexOfGrid[c], vertexOfGrid[d], threhold); \
	}

//marching cube主体函数实现，是对体素格点做的
__global__ void marchingCubes(int *count, const Device::Point *grids, const int3 dim, Device::TrianglePN *triList, short *triListSize, double threhold, int *edgeTableGPU, int *triTableGPU)
{
	int3 DIRECTION[8] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
	unsigned int insideIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	unsigned int outsideIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
	unsigned int index = insideIndex + outsideIndex * blockSize;
	int z = index / (dim.x * dim.y);
	int y = (index % (dim.x * dim.y)) / dim.x;
	int x = (index % (dim.x * dim.y)) % dim.x;
	int3 gridPos = make_int3(x, y, z);

	if (gridPos.x < (dim.x - 1) && gridPos.y < (dim.y - 1) && gridPos.z < (dim.z - 1))
	{

		Device::Point vertexOnEdge[12];
		Device::Point vertexOfGrid[8];
		int cubeIndex = 0;

		for (int i = 0; i < 8; ++i)
		{
			int3 tempPos = gridPos + DIRECTION[i];
			vertexOfGrid[i] = grids[tempPos.z * dim.x * dim.y + tempPos.y * dim.x + tempPos.x];
		}
		// 检查顶点权值
		INDEX_BUILD(0, 1);
		INDEX_BUILD(1, 2);
		INDEX_BUILD(2, 4);
		INDEX_BUILD(3, 8);
		INDEX_BUILD(4, 16);
		INDEX_BUILD(5, 32);
		INDEX_BUILD(6, 64);
		INDEX_BUILD(7, 128);

		if (edgeTableGPU[cubeIndex] == 0)
		{
			return;
		}
		atomicAdd(count, 1);
		// 求解对应边上的交点
		VERTEX_LIST_BUILD(0, 1, 0, 1);
		VERTEX_LIST_BUILD(1, 2, 1, 2);
		VERTEX_LIST_BUILD(2, 4, 2, 3);
		VERTEX_LIST_BUILD(3, 8, 3, 0);
		VERTEX_LIST_BUILD(4, 16, 4, 5);
		VERTEX_LIST_BUILD(5, 32, 5, 6);
		VERTEX_LIST_BUILD(6, 64, 6, 7);
		VERTEX_LIST_BUILD(7, 128, 7, 4);
		VERTEX_LIST_BUILD(8, 256, 0, 4);
		VERTEX_LIST_BUILD(9, 512, 1, 5);
		VERTEX_LIST_BUILD(10, 1024, 2, 6);
		VERTEX_LIST_BUILD(11, 2048, 3, 7);

		for (int i = 0; triTableGPU[cubeIndex * 16 + i] != -1; i += 3)
		{
			triList[index * 5 + triListSize[index]] = Device::TrianglePN(
				Device::Triangle(vertexOnEdge[triTableGPU[cubeIndex * 16 + i]].pos,
								 vertexOnEdge[triTableGPU[cubeIndex * 16 + i + 1]].pos,
								 vertexOnEdge[triTableGPU[cubeIndex * 16 + i + 2]].pos),
				Device::Triangle(vertexOnEdge[triTableGPU[cubeIndex * 16 + i]].normal,
								 vertexOnEdge[triTableGPU[cubeIndex * 16 + i + 1]].normal,
								 vertexOnEdge[triTableGPU[cubeIndex * 16 + i + 2]].normal));
			triListSize[index]++;
		}
	}
}

void kernel_wrapper_3(bool *voxelThinned, Device::Point *voxelPointMap, const int &dimX, const int &dimY, const int &dimZ, float &originX, float &originY, float &originZ, float scale, float threhold, std::vector<Triangle> &triangleVector)
{

	int *count = new int(0);
	int *gpu_count;
	gpuErrchk(cudaMalloc((void **)&gpu_count, sizeof(int)));
	gpuErrchk(cudaMemcpy(gpu_count, count, sizeof(int), cudaMemcpyHostToDevice));

	dim3 Dg(ceil(dimX / 8.0), ceil(dimY / 8.0), ceil(dimZ / 8.0));
	dim3 Db(8, 8, 8);
	// 求解最近的几个个点，确定切面及各点到它的距离作为权值给格点
	bool *gpu_voxel_thinned;
	gpuErrchk(cudaMalloc((void **)&gpu_voxel_thinned, sizeof(bool) * dimX * dimY * dimZ));
	gpuErrchk(cudaMemcpy(gpu_voxel_thinned, voxelThinned, sizeof(bool) * dimX * dimY * dimZ, cudaMemcpyHostToDevice));
	int3 dim = make_int3(dimX, dimY, dimZ);
	float3 orig = make_float3(originX, originY, originZ);
	Device::Point *gpu_grids;
	gpuErrchk(cudaMalloc((void **)&gpu_grids, sizeof(Device::Point) * dimX * dimY * dimZ));
	Device::Point *gpu_voxel_point_map;
	gpuErrchk(cudaMalloc((void **)&gpu_voxel_point_map, sizeof(Device::Point) * dimX * dimY * dimZ));
	gpuErrchk(cudaMemcpy(gpu_voxel_point_map, voxelPointMap, sizeof(Device::Point) * dimX * dimY * dimZ, cudaMemcpyHostToDevice));

	gridBuilder<<<Dg, Db>>>(gpu_count, gpu_grids, gpu_voxel_point_map, scale, orig, dim, gpu_voxel_thinned);
	gpuErrchk(cudaMemcpy(count, gpu_count, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << *count << std::endl;
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	std::cout << "格点处理完毕" << std::endl;

	gpuErrchk(cudaFree(gpu_voxel_thinned));

	gpuErrchk(cudaFree(gpu_voxel_point_map));

	short *triangle_num = new short[dimX * dimY * dimZ]();
	short *gpu_triangle_num;
	gpuErrchk(cudaMalloc((void **)&gpu_triangle_num, sizeof(short) * dimX * dimY * dimZ));
	gpuErrchk(cudaMemcpy(gpu_triangle_num, triangle_num, sizeof(short) * dimX * dimY * dimZ, cudaMemcpyHostToDevice));

	Device::TrianglePN *triangle_list;
	Device::TrianglePN *gpu_triangle_list;
	int estimateSize = dimX * dimY * dimZ * 5; // 每个线程最多会算出来五个三角形
	gpuErrchk(cudaMalloc((void **)&gpu_triangle_list, sizeof(Device::TrianglePN) * estimateSize));
	// 由于marching cube中的getGrid函数，这里要少一层
	int *gpu_tri_table, *gpu_edge_table;
	gpuErrchk(cudaMalloc((void **)&gpu_tri_table, sizeof(int) * 4096));
	gpuErrchk(cudaMalloc((void **)&gpu_edge_table, sizeof(int) * 256));
	gpuErrchk(cudaMemcpy(gpu_tri_table, triTable, sizeof(int) * 4096, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(gpu_edge_table, edgeTable, sizeof(int) * 256, cudaMemcpyHostToDevice));

	*count = 0;
	gpuErrchk(cudaMemcpy(gpu_count, count, sizeof(int), cudaMemcpyHostToDevice));
	marchingCubes<<<Dg, Db>>>(gpu_count, gpu_grids, dim, gpu_triangle_list, gpu_triangle_num, threhold, gpu_edge_table, gpu_tri_table);
	gpuErrchk(cudaMemcpy(count, gpu_count, sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << *count << std::endl;

	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaMemcpy(triangle_num, gpu_triangle_num, sizeof(short) * dimX * dimY * dimZ, cudaMemcpyDeviceToHost));
	triangle_list = new Device::TrianglePN[estimateSize];
	gpuErrchk(cudaMemcpy(triangle_list, gpu_triangle_list, sizeof(Device::TrianglePN) * estimateSize, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaFree(gpu_triangle_list));
	gpuErrchk(cudaFree(gpu_triangle_num));
	gpuErrchk(cudaFree(gpu_grids));

	// 得到了三角形triangleList
	int realTriangleNum = 0;
	for (int i = 0; i < dimX * dimY * dimZ; ++i)
	{
		realTriangleNum += triangle_num[i];
	}

	triangleVector.resize(realTriangleNum);
	std::cout << "三角形面片数量" << realTriangleNum << std::endl;

	int tempIndex = 0;
	for (int i = 0; i < dimX * dimY * dimZ; ++i)
	{
		for (int j = 0; j < triangle_num[i]; ++j)
		{
			triangleVector[tempIndex].pos.m_v1.m_x = triangle_list[i * 5 + j].pos.m_v1.x;
			triangleVector[tempIndex].pos.m_v1.m_y = triangle_list[i * 5 + j].pos.m_v1.y;
			triangleVector[tempIndex].pos.m_v1.m_z = triangle_list[i * 5 + j].pos.m_v1.z;

			triangleVector[tempIndex].pos.m_v2.m_x = triangle_list[i * 5 + j].pos.m_v2.x;
			triangleVector[tempIndex].pos.m_v2.m_y = triangle_list[i * 5 + j].pos.m_v2.y;
			triangleVector[tempIndex].pos.m_v2.m_z = triangle_list[i * 5 + j].pos.m_v2.z;

			triangleVector[tempIndex].pos.m_v3.m_x = triangle_list[i * 5 + j].pos.m_v3.x;
			triangleVector[tempIndex].pos.m_v3.m_y = triangle_list[i * 5 + j].pos.m_v3.y;
			triangleVector[tempIndex].pos.m_v3.m_z = triangle_list[i * 5 + j].pos.m_v3.z;

			triangleVector[tempIndex].normal.m_v1.m_x = triangle_list[i * 5 + j].normal.m_v1.x;
			triangleVector[tempIndex].normal.m_v1.m_y = triangle_list[i * 5 + j].normal.m_v1.y;
			triangleVector[tempIndex].normal.m_v1.m_z = triangle_list[i * 5 + j].normal.m_v1.z;

			triangleVector[tempIndex].normal.m_v2.m_x = triangle_list[i * 5 + j].normal.m_v2.x;
			triangleVector[tempIndex].normal.m_v2.m_y = triangle_list[i * 5 + j].normal.m_v2.y;
			triangleVector[tempIndex].normal.m_v2.m_z = triangle_list[i * 5 + j].normal.m_v2.z;

			triangleVector[tempIndex].normal.m_v3.m_x = triangle_list[i * 5 + j].normal.m_v3.x;
			triangleVector[tempIndex].normal.m_v3.m_y = triangle_list[i * 5 + j].normal.m_v3.y;
			triangleVector[tempIndex].normal.m_v3.m_z = triangle_list[i * 5 + j].normal.m_v3.z;
			tempIndex++;
		}
	}
}