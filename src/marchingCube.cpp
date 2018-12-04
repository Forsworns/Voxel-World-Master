#include "includes/CompFab.h"
#include "includes/marchingTable.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "includes/cuda_math.h"
using namespace std;

#define USE_THIN
#define FOUT_LINE(a) fout<<a<<endl;
#define FOUT_POINT(a,b,c) fout<<a<<" "<<b<<" "<<c<<endl;
#define NONE_ZERO 0.0001f

namespace Device {
	struct Point
	{
		float3 pos;
		float3 normal;
		double weight;
		Point() :pos(make_float3(0, 0, 0)), normal(make_float3(0, 0, 0)), weight(0) {}
		Point(float3 p, float3 n, double w) : pos(p), normal(n), weight(w) {}
	};
}

Point ***grids;
const int INF = 10000;
Point *voxelPointMap;
const CompFab::Vec3 NEIGHBOR[8] = { {-1, -1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 0}, {-1, -1, -1}, {-1, 0, -1}, {0, -1, -1}, {0, 0, -1} };
const CompFab::Vec3 DIRECTION[8] = { {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1} };

// ���������е㷨����
void estimateNormal(Point &point)
{
	// ����matlab�е�ʵ�֣�
}

Point &getGrid(const CompFab::Vec3 &pos)
{
	return grids[int(pos.m_x)][int(pos.m_y)][int(pos.m_z)];
}

// ���Բ�ֵ����
Point interpolate(const Point &p1, const Point &p2, const float &threhold)
{
	Point point;
	if (abs(threhold - p1.weight) < NONE_ZERO) {
		return p1;
	}
	if (abs(threhold - p2.weight) < NONE_ZERO) {
		return p2;
	}
	if (abs(p2.weight - p1.weight) < NONE_ZERO) {
		return p1;
	}
	float weight = (threhold - p1.weight) / (p2.weight - p1.weight);

	// float weight = 0.5;
	point.pos = p1.pos + weight * (p2.pos - p1.pos);
	point.normal = p1.normal + weight * (p2.normal - p1.normal);
	point.normal.normalize();
	return point;
}

// �������ظ���Ȩֵ
void gridBuilder(CompFab::Vec3 pos, double scale, CompFab::Vec3 orig, CompFab::Vec3 dim, bool *voxel)
{
	Point *grid = &getGrid(pos);
	grid->pos = CompFab::Vec3(pos.m_x * scale + orig.m_x, pos.m_y * scale + orig.m_y, pos.m_z * scale + orig.m_z);

	int neighborNum = 0;
	CompFab::Vec3 neighborPos = { 0, 0, 0 };
	CompFab::Vec3 neighborNormal = { 0, 0, 0 };
	// Ӧ�ø�������ļ��������weight������������һ��İ˸��㣺����֮ǰvoxel����Ϣ��ֱ���Ҷ��㸽���İ˸��ھӣ�ע��Ӧ���Ƕ����Ӧ���ص�Ͷ�����Ҫ���ھ����ص�֮��Ĺ�ϵ
	for (int i = 0; i < 8; ++i)
	{
		CompFab::Vec3 tempPos = pos + NEIGHBOR[i];
		if (tempPos.m_x < 0 || tempPos.m_y < 0 || tempPos.m_z < 0 ||
			tempPos.m_x >= dim.m_x || tempPos.m_y >= dim.m_y || tempPos.m_z >= dim.m_z ||
			!voxel[int(tempPos.m_z * dim.m_x * dim.m_y + tempPos.m_y * dim.m_x + tempPos.m_x)])
		{
			continue;
		}
		Point tempPoint = voxelPointMap[int(tempPos.m_z * dim.m_x * dim.m_y + tempPos.m_y * dim.m_x + tempPos.m_x)];
		neighborPos = neighborPos + tempPoint.pos;
		neighborNormal = neighborNormal + tempPoint.normal;
		neighborNum++;
	}
	// ������ھ���voxelʱ��û�����Ĳ��֣�������
	if (neighborNum == 0)
	{
		grid->weight = INF;
		return;
	}
	else
	{
		neighborNormal.normalize();
		neighborPos = (1.0f / neighborNum) * neighborPos;
		grid->weight = (grid->pos - neighborPos) * neighborNormal;
	}
}

#define INDEX_BUILD(a, b)                  \
    if (vertexOfGrid[a].weight < threhold) \
    {                                      \
        cubeIndex |= b;                    \
    }

#define VERTEX_LIST_BUILD(a, b, c, d)                                              \
    if (edgeTable[cubeIndex] & b)                                                  \
    {                                                                              \
        vertexOnEdge[a] = interpolate(vertexOfGrid[c], vertexOfGrid[d], threhold); \
    }

//marching cube���庯��ʵ�֣��Ƕ����ظ������
void marchingCubes(const CompFab::Vec3 &gridPos, vector<Triangle> &triList, double threhold)
{
	Point vertexOnEdge[12];
	Point vertexOfGrid[8];
	int cubeIndex = 0;

	for (int i = 0; i < 8; ++i)
	{
		vertexOfGrid[i] = getGrid(gridPos + DIRECTION[i]);
	}

	// ��鶥��Ȩֵ
	INDEX_BUILD(0, 1);
	INDEX_BUILD(1, 2);
	INDEX_BUILD(2, 4);
	INDEX_BUILD(3, 8);
	INDEX_BUILD(4, 16);
	INDEX_BUILD(5, 32);
	INDEX_BUILD(6, 64);
	INDEX_BUILD(7, 128);

	if (edgeTable[cubeIndex] == 0)
	{
		return;
	}

	// ����Ӧ���ϵĽ���
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


	for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
	{
		triList.emplace_back(Triangle(
			CompFab::Triangle(vertexOnEdge[triTable[cubeIndex][i]].pos,
				vertexOnEdge[triTable[cubeIndex][i + 1]].pos,
				vertexOnEdge[triTable[cubeIndex][i + 2]].pos),
			CompFab::Triangle(vertexOnEdge[triTable[cubeIndex][i]].normal,
				vertexOnEdge[triTable[cubeIndex][i + 1]].normal,
				vertexOnEdge[triTable[cubeIndex][i + 2]].normal)
		)
		);
	}
}

void prepare(bool useGPU, Device::Point *result, const long long int &blockNum, bool *voxel, bool *voxelThined,const int &dimX,const int &dimY,const int&dimZ,vector<Point> &points,float &originX,float &originY,float &originZ,float scale,float threhold, vector<Triangle> &triangleVector) {
	ifstream fin;
	voxelPointMap = new Point[blockNum];
	long long int x, y, z;
	// �Դ��������Ҫע��һ�£����ܵø�
	for (long long int i = 0; i < blockNum; ++i)
	{
#ifdef USE_THIN
		if (voxelThined[i])
#else
		if (voxel[i])
#endif
		{
			z = i / (dimX * dimY);
			long long int xy = i % (dimX * dimY);
			y = xy / dimY;
			x = xy % dimY;
			float posX, posY, posZ;
			posX = (x + 0.5) * scale + originX;
			posY = (y + 0.5) * scale + originY;
			posZ = (z + 0.5) * scale + originZ;
			points.emplace_back(Point(CompFab::Vec3(posX, posY, posZ), CompFab::Vec3(x, y, z), 0)); // ������Ϊ��֮�󷽱���±�������
			voxelPointMap[int(z*dimX*dimY + y * dimX + x)] = Point(CompFab::Vec3(posX, posY, posZ), CompFab::Vec3(0, 0, 0), 0);
		}
	}
	cout << "������" << points.size() << "����" << endl;
	// ����˵��ƴ�����points��

	//�������
	ofstream fout("test_32.ply");
	FOUT_LINE("ply");
	FOUT_LINE("format ascii 1.0");
	FOUT_POINT("element", "vertex", points.size());
	FOUT_LINE("property float x");
	FOUT_LINE("property float y");
	FOUT_LINE("property float z");
	FOUT_LINE("end_header");
	for (auto &x : points) {
		FOUT_POINT(x.pos.m_x, x.pos.m_y, x.pos.m_z);
	}
	fout.close();

	// ���Ƹ��㷨����
	for (auto &x : points)
	{
		estimateNormal(x);
	}

	fin.open("normals.txt");
	vector<Point>::iterator it = points.begin();
	while (!fin.eof()) {
		if (it < points.end()) {
			char temp;
			double tempNum;
			fin >> temp;
			CompFab::Vec3* p = &(voxelPointMap[int(it->normal.m_z*dimX*dimY + it->normal.m_y*dimX + it->normal.m_x)].normal);
			fin >> it->normal.m_x >> temp >> it->normal.m_y >> temp >> it->normal.m_z;
			p->m_x = it->normal.m_x;
			p->m_y = it->normal.m_y;
			p->m_z = it->normal.m_z;
			fin >> temp;
			fin >> tempNum;
			fin >> temp;
			++it;
		}
		else {
			break;
		}
	}
	fin.close();
	cout << "�������������" << endl;

	grids = new Point**[dimX + 1];
	for (int i = 0; i < dimX + 1; ++i) {
		grids[i] = new Point*[dimY + 1];
		for (int j = 0; j < dimY + 1; ++j) {
			grids[i][j] = new Point[dimZ + 1];
		}
	}

	if (useGPU) {;
		for (int i = 0; i < blockNum; ++i) {
			result[i].normal = make_float3(voxelPointMap[i].normal.m_x, voxelPointMap[i].normal.m_y, voxelPointMap[i].normal.m_z);
			result[i].pos = make_float3(voxelPointMap[i].pos.m_x, voxelPointMap[i].pos.m_y, voxelPointMap[i].pos.m_z);
		}
	}
}

void useCpuToUnvoxelize(bool *voxel, bool *voxelThined, const int &dimX, const int &dimY, const int&dimZ, float &originX, float &originY, float &originZ, float scale, float threhold, vector<Triangle> &triangleVector) {
	// �������ļ������㣬ȷ�����漰���㵽���ľ�����ΪȨֵ�����
	for (int i = 0; i < dimX + 1; ++i) {
		for (int j = 0; j < dimY + 1; ++j) {
			for (int k = 0; k < dimZ + 1; ++k) {
				CompFab::Vec3 gridPos(i, j, k);
				CompFab::Vec3 dim(dimX, dimY, dimZ);
				CompFab::Vec3 orig(originX, originY, originZ);
#ifdef USE_THIN
				gridBuilder(gridPos, scale, orig, dim, voxelThined);
#else
				gridBuilder(gridPos, scale, orig, dim, voxel);
#endif
			}
		}
	}
	cout << "��㴦�����" << endl;

	int triangleNum = 0;
	// ����marching cube�е�getGrid����������Ҫ��һ��
	for (int i = 0; i < dimX; ++i) {
		for (int j = 0; j < dimY; ++j) {
			for (int k = 0; k < dimZ; ++k) {
				CompFab::Vec3 gridPos(i, j, k);
				marchingCubes(gridPos, triangleVector, threhold);
			}
		}
	}
	// �õ���������triangleVector
	cout << "��������Ƭ����" << triangleVector.size() << endl;

}

void outputToPly(string fileName, vector<Triangle> &triangleVector) {
	// ply��ʽ���
	ofstream fout;
	fout.open(fileName);
	FOUT_LINE("ply");
	FOUT_LINE("format ascii 1.0");
	FOUT_POINT("element", "vertex", triangleVector.size() * 3);
	FOUT_LINE("property float x");
	FOUT_LINE("property float y");
	FOUT_LINE("property float z");
	FOUT_POINT("element", "face", triangleVector.size());
	FOUT_LINE("property list uchar int vertex_index");
	FOUT_LINE("end_header");
	for (int i = 0; i < triangleVector.size(); ++i) {
		FOUT_POINT(triangleVector[i].pos.m_v1.m_x, triangleVector[i].pos.m_v1.m_y, triangleVector[i].pos.m_v1.m_z);
		FOUT_POINT(triangleVector[i].pos.m_v2.m_x, triangleVector[i].pos.m_v2.m_y, triangleVector[i].pos.m_v2.m_z);
		FOUT_POINT(triangleVector[i].pos.m_v3.m_x, triangleVector[i].pos.m_v3.m_y, triangleVector[i].pos.m_v3.m_z);
	}
	for (int i = 0; i < triangleVector.size() * 3;) {
		fout << 3 << " ";
		FOUT_POINT(i++, i++, i++);//���ع��췴��Ĺ�����
	}
	fout.close();
}