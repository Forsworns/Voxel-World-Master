#include "includes/CompFab.h"
#include "math.h"
#include <iostream>
#include <string>
#include <sstream>
#include "stdio.h"
#include <vector>
#include <random>
#include "includes/vec3.h"
using namespace std;

#define EPSILONF 0.000001
#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

// generate a random float in (0,1)
int generateRadom(default_random_engine generator, uniform_real_distribution<float> distribution)
{
    return distribution(generator);
}

bool isInside(unsigned int numIntersections, bool double_thick)
{
    //double指代双层的模型，这里判断内外用的算法是类似平面点是否在内部
    if (double_thick)
        return (numIntersections / 2) % 2 == 1;
    else
        return numIntersections % 2 == 1;
}

// adapted from: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
bool intersectsThisTri(CompFab::Triangle &triangle, vec3<float> dir, vec3<float> pos)
{
    // 与每一个三角形面片判断是否相交
    vec3<float> V1(triangle.m_v1.m_x, triangle.m_v1.m_y, triangle.m_v1.m_z);
    vec3<float> V2(triangle.m_v2.m_x, triangle.m_v2.m_y, triangle.m_v2.m_z);
    vec3<float> V3(triangle.m_v3.m_x, triangle.m_v3.m_y, triangle.m_v3.m_z);

    //Find vectors for two edges sharing V1
    vec3<float> e1 = V2 - V1;
    vec3<float> e2 = V3 - V1;

    //Begin calculating determinant - also used to calculate u parameter dir确定一个采样/观测的方向
    vec3<float> P = cross(dir, e2);

    //if determinant is near zero, ray lies in plane of triangle 即两条共面直线做叉积成了法向量，再做点积成了0
    float det = dot(e1, P);

    //NOT CULLING
    if (det > -EPSILONF && det < EPSILONF)
        return false;
    float inv_det = 1.f / det;

    // calculate distance from V1 to ray origin
    vec3<float> T = pos - V1;
    //Calculate u parameter and test bound
    float u = dot(T, P) * inv_det;
    //The intersection lies outside of the triangle
    if (u < 0.f || u > 1.f)
        return false;

    //Prepare to test v parameter
    vec3<float> Q = cross(T, e1);
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

void useCpuToVoxelize(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick)
{
	int numTriangles = triangles.size();
	float spacing = (float)g_voxelGrid->m_spacing; // information about the scale
	vec3<float> lower_left(g_voxelGrid->m_lowerLeft.m_x, g_voxelGrid->m_lowerLeft.m_y, g_voxelGrid->m_lowerLeft.m_z); // the start of the model
    default_random_engine generator;
    uniform_real_distribution<float> distribution(0.0, 1.0);
    g_voxelGrid->m_insideArray = new bool[w * h * d];

    if (samples > 0)
    {
        for (int l = 0; l < samples; ++l)
        {
            for (int i = 0; i < w; ++i)
            {
                for (int j = 0; j < h; ++j)
                {
                    for (int k = 0; k < d; ++k)
                    {
                        // find linearlized index in final boolean array
                        unsigned int index_out = k * (w * h) + j * h + i;
                        // find world space position of the voxel
                        vec3<float> pos(lower_left.get_x() + spacing * i, lower_left.get_y() + spacing * j, lower_left.get_z() + spacing * k);
                        vec3<float> dir;

                        // we will randomly sample 3D space by sending rays in randomized directions
                        int votes = 0;
                        float theta;
                        float z;

                        for (int sample = 0; sample < samples; ++sample)
                        {
                            // compute the random direction. Convert from polar to euclidean to get an even distribution
                            theta = generateRadom(generator, distribution) * 2.f * E_PI;
                            z = generateRadom(generator, distribution) * 2.f - 1.f;

                            dir.set_x(sqrt(1 - z * z) * cosf(theta));
                            dir.set_y(sqrt(1 - z * z) * sinf(theta));
                            dir.set_z(sqrt(1 - z * z) * cosf(theta));

                            // check if the voxel is inside of the mesh.
                            // if it is inside, then there should be an odd number of
                            // intersections with the surrounding mesh
                            unsigned int intersections = 0;
                            for (int num = 0; num < numTriangles; ++num)
                                if (intersectsThisTri(triangles[num], dir, pos))
                                    intersections += 1;
                            if (isInside(intersections, double_thick))
                                votes += 1;
                        }
                        // choose the most popular answer from all of the randomized samples
                        g_voxelGrid->m_insideArray[index_out] = votes >= (samples / 2.f);
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < w; ++i)
        {
            for (int j = 0; j < h; ++j)
            {
                for (int k = 0; k < d; ++k)
                {                                                                               
                    // pick an arbitrary sampling direction
                    vec3<float> dir(1.0, 0.0, 0.0);
                    // find linearlized index in final boolean array 数组元素取址
                    unsigned int index_out = k * (w * h) + j * h + i;

                    // find world space position of the voxel
                    vec3<float> pos(lower_left.get_x() + spacing * i, lower_left.get_y() + spacing * j, lower_left.get_z() + spacing * k);

                    // check if the voxel is inside of the mesh.
                    // if it is inside, then there should be an odd number of
                    // intersections with the surrounding mesh
                    unsigned int intersections = 0;
                    for (int num = 0; num < numTriangles; ++num)
                        if (intersectsThisTri(triangles[num], dir, pos))
                            intersections += 1;

                    // store answer
                    g_voxelGrid->m_insideArray[index_out] = isInside(intersections, double_thick);
                }
            }
        }
    }
}