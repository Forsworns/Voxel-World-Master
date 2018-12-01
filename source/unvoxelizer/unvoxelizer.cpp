#include "CompFab.h"
#include "marchingTable.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
using namespace std;

#define USE_THIN
#define FOUT_LINE(a) fout<<a<<endl;
#define FOUT_POINT(a,b,c) fout<<a<<" "<<b<<" "<<c<<endl;
#define NONE_ZERO 0.0001f


struct Triangle
{
	CompFab::Triangle pos;
	CompFab::Triangle normal;
	Triangle() {}
	Triangle(CompFab::Triangle p, CompFab::Triangle n) : pos(p), normal(n) {}
};

struct Point
{
	CompFab::Vec3 pos;
	CompFab::Vec3 normal;
	double weight;
	Point() :pos(CompFab::Vec3(0, 0, 0)), normal(CompFab::Vec3(0, 0, 0)), weight(0) {}
	Point(CompFab::Vec3 p, CompFab::Vec3 n, double w) : pos(p), normal(n), weight(w) {}
};

Point ***grids;
const int INF = 10000;
// 这个map可以改成数组，把vec3坐标改成数组就行了（往gpu移动的话
Point *voxelPointMap;
const CompFab::Vec3 NEIGHBOR[8] = { {-1, -1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 0}, {-1, -1, -1}, {-1, 0, -1}, {0, -1, -1}, {0, 0, -1} };
const CompFab::Vec3 DIRECTION[8] = { {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1} };

// 求解各点云中点法向量
void estimateNormal(Point &point)
{
	// 仿照matlab中的实现？
}

Point &getGrid(const CompFab::Vec3 &pos)
{
	return grids[int(pos.m_x)][int(pos.m_y)][int(pos.m_z)];
}

// 线性插值函数
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

// 求解各体素格点的权值
void gridBuilder(CompFab::Vec3 pos, double scale, CompFab::Vec3 orig, CompFab::Vec3 dim, bool *voxel)
{
	Point *grid = &getGrid(pos);
	grid->pos = CompFab::Vec3(pos.m_x * scale + orig.m_x, pos.m_y * scale + orig.m_y, pos.m_z * scale + orig.m_z);

	int neighborNum = 0;
	CompFab::Vec3 neighborPos = { 0, 0, 0 };
	CompFab::Vec3 neighborNormal = { 0, 0, 0 };
	// 应该根据最近的几个点计算weight。这里改用最近一层的八个点：利用之前voxel的信息，直接找顶点附近的八个邻居，注意应该是顶点对应体素点和顶点想要的邻居体素点之间的关系
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
	// 如果是邻居在voxel时是没填满的部分，则跳过
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

//marching cube主体函数实现，是对体素格点做的
void marchingCubes(const CompFab::Vec3 &gridPos, vector<Triangle> &triList, double threhold)
{
	Point vertexOnEdge[12];
	Point vertexOfGrid[8];
	int cubeIndex = 0;

	for (int i = 0; i < 8; ++i)
	{
		vertexOfGrid[i] = getGrid(gridPos + DIRECTION[i]);
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

	if (edgeTable[cubeIndex] == 0)
	{
		return;
	}

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

int func_nc8_CPU(int *b);
void HilditchThin_CPU(bool* inside_array, bool* output_array, int w, int h, int d);

enum ModelType
{
	BINVOX,
	PCD
};



int main()
{
	ifstream fin;
	string fileName;
	ModelType modeType = PCD;
	vector<Point> points;
	float originX, originY, originZ;
	double scale;
	int dimX, dimY, dimZ;
	float threhold = 0;
	vector<Triangle> triangleVector;
	cout << "最大" << triangleVector.max_size() << endl;

	bool *voxel, *voxelThined;
	long long int blockNum;

	switch (modeType)
	{
	case BINVOX:
	{
		char title[12];
		fileName = "bone1_64.binvox";
		fin.open(fileName, ios::binary);
		if (!fin)
		{
			cout << "未能打开文件" << endl;
			return 1;
		}
		fin.getline(title, 12);
		cout << title << endl;
		fin >> title;
		fin >> dimX >> dimY >> dimZ;
		cout << title << " " << dimX << " " << dimY << " " << dimZ << endl;
		fin >> title;
		fin >> originX >> originY >> originZ;
		cout << title << " " << originX << " " << originY << " " << originZ << endl;

		fin >> title;
		fin >> scale;
		cout << title << " " << scale << endl;
		fin.get();
		fin.getline(title, 12);
		cout << title << endl;


		// binvox的格式是先是一个值0或1，然后是有多少个连续的相同的值
		vector<int> binvox;
		while (!fin.eof())
		{
			char temp;
			fin.read(&temp, sizeof(char));
			int vec = temp - '\0';
			if (vec < 0)
			{
				binvox.emplace_back(vec + 256);
			}
			else
			{
				binvox.emplace_back(vec);
			}
		}
		cout << binvox.size() << endl;
		fin.close();

		// 恢复binvox输出时的数组
		int index = 0;
		blockNum = dimX * dimY * dimZ;
		voxel = new bool[blockNum]();
		for (vector<int>::iterator it = binvox.begin(); it < binvox.end(); ++it)
		{
			bool isInside = bool(*it);
			++it;
			if (it == binvox.end())
			{
				break;
			}
			int isInsideNum = *it;
			for (int i = 0; i < isInsideNum; ++i)
			{
				voxel[index++] = isInside;
			}
		}

		break;
	}
	case PCD:
	{
		fileName = "L.pcd";
		dimX = 256;
		dimY = 256;
		dimZ = 256;
		vector<CompFab::Vec3> pointCloud;
		fin.open(fileName);
		char title[80];
		do {
			fin.getline(title, 80);
			cout << title << endl;
		} while (strcmp(title,"DATA ascii")!=0);

		while (!fin.eof()) {
			float x, y, z;
			fin >> x >> y >> z;
			pointCloud.emplace_back(CompFab::Vec3(x,y,z));
		}
		
		fin.close();

		// 计算origin(左下角)
		float xMin=INF, yMin = INF, zMin = INF, xMax = -INF, yMax = -INF, zMax = -INF;
		for (vector<CompFab::Vec3>::iterator it = pointCloud.begin(); it < pointCloud.end(); ++it) {
			xMin = xMin < it->m_x ? xMin : it->m_x;
			yMin = yMin < it->m_y ? yMin : it->m_y;
			zMin = zMin < it->m_z ? zMin : it->m_z;
			xMax = xMax > it->m_x ? xMax : it->m_x;
			yMax = yMax > it->m_y ? yMax : it->m_y;
			zMax = zMax > it->m_z ? zMax : it->m_z;
		}
		originX=xMin, originY = yMin, originZ = zMin;

		// 计算scale
		if (xMax-xMin>yMax- yMin&& yMax - yMin > zMax - zMin) {
			scale = (xMax - xMin) / (dimX-1);
		}
		else if(xMax - xMin < yMax - yMin && yMax - yMin > zMax - zMin) {
			scale = (yMax - yMin) / (dimY-1);
		}
		else {
			scale = (zMax - zMin) / (dimZ-1);
		}

		// 将点云映射到体素空间得到voxel数组z,y,x
		blockNum = dimX * dimY * dimZ;
		voxel = new bool[blockNum]();
		for (vector<CompFab::Vec3>::iterator it = pointCloud.begin(); it < pointCloud.end();++it) {
			int z = floor((it->m_z-originZ) / scale);
			int y = floor((it->m_y-originY) / scale);
			int x = floor((it->m_x-originX) / scale);
			voxel[z*dimX*dimY + y * dimX + x] = true;
		}

		break;
	}
	default:
		return -2;
	}


	// 细化处理
	voxelThined = new bool[blockNum]();
	HilditchThin_CPU(voxel, voxelThined, dimX, dimY, dimZ);

	voxelPointMap = new Point[blockNum];
	long long int x, y, z;
	// 对大数的情况要注意一下，可能得改
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
			points.emplace_back(Point(CompFab::Vec3(posX, posY, posZ), CompFab::Vec3(x, y, z), 0)); // 这里是为了之后方便更新表中数据
			voxelPointMap[int(z*dimX*dimY + y * dimX + x)] = Point(CompFab::Vec3(posX, posY, posZ), CompFab::Vec3(0, 0, 0), 0);
		}
	}
	cout << "处理了" << points.size() << "个点" << endl;
	// 求出了点云存在了points中

	//输出点云
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

	// 估计各点法向量
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
	cout << "法向量处理完毕" << endl;

	grids = new Point**[dimX + 1];
	for (int i = 0; i < dimX + 1; ++i) {
		grids[i] = new Point*[dimY + 1];
		for (int j = 0; j < dimY + 1; ++j) {
			grids[i][j] = new Point[dimZ + 1];
		}
	}



	// 求解最近的几个个点，确定切面及各点到它的距离作为权值给格点
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
	cout << "格点处理完毕" << endl;

	int triangleNum = 0;
	// 由于marching cube中的getGrid函数，这里要少一层
	for (int i = 0; i < dimX; ++i) {
		for (int j = 0; j < dimY; ++j) {
			for (int k = 0; k < dimZ; ++k) {
				CompFab::Vec3 gridPos(i, j, k);
				marchingCubes(gridPos, triangleVector, threhold);
			}
		}
	}
	// 得到了三角形triangleVector
	cout << "三角形面片数量" << triangleVector.size() << endl;

	// ply格式输出
	fout.open("output.ply");
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
		FOUT_POINT(i++, i++, i++);//不必构造反向的关联表
	}
	fout.close();

	return 0;
}


int func_nc8_CPU(int *b)
//端点的连通性检测
{
	int n_odd[4] = { 1, 3, 5, 7 };  //四邻域
	int i, j, sum, d[10];

	for (i = 0; i <= 9; i++) {
		j = i;
		if (i == 9) j = 1;
		if (abs(*(b + j)) == 1)
		{
			d[i] = 1;
		}
		else
		{
			d[i] = 0;
		}
	}
	sum = 0;
	for (i = 0; i < 4; i++)
	{
		j = n_odd[i];
		sum = sum + d[j] - d[j] * d[j + 1] * d[j + 2];
	}
	return (sum);
}

//细化算法
void HilditchThin_CPU(bool* inside_array, bool* output_array, int w, int h, int d)
{
	int GRAY = 128, WHITE = 255, BLACK = 0;
	for (int z = 0; z < d; ++z)
	{
		int *img = new int[w * h];
		for (int i = 0; i < h; i++)
		{

			for (int j = 0; j < w; j++)
			{
				img[j + i * w] = (int)inside_array[j + i * w + z * w * h] * 255;
			}
		}
		int offset[9][2] = { {0,0},{1,0},{1,-1},{0,-1},{-1,-1},
	{-1,0},{-1,1},{0,1},{1,1} };
		//四邻域的偏移量
		int n_odd[4] = { 1, 3, 5, 7 };
		int px, py;
		int b[9];                      //3*3格子的灰度信息
		int condition[6];              //1-6个条件是否满足
		int counter;                   //移去像素的数量
		int i, x, y, copy, sum;
		do
		{

			counter = 0;

			for (y = 0; y < h; y++)
			{

				for (x = 0; x < w; x++)
				{

					//前面标记为删除的像素，我们置其相应邻域值为-1
					for (i = 0; i < 9; i++)
					{
						b[i] = 0;
						px = x + offset[i][0];
						py = y + offset[i][1];
						if (px >= 0 && px < w &&    py >= 0 && py < h)
						{
							// printf("%d\n", img[py*step+px]);
							if (img[py*w + px] == WHITE)
							{
								b[i] = 1;
							}
							else if (img[py*w + px] == GRAY)
							{
								b[i] = -1;
							}
						}
					}
					for (i = 0; i < 6; i++)
					{
						condition[i] = 0;
					}

					//条件1，是前景点
					if (b[0] == 1) condition[0] = 1;

					//条件2，是边界点
					sum = 0;
					for (i = 0; i < 4; i++)
					{
						sum = sum + 1 - abs(b[n_odd[i]]);
					}
					if (sum >= 1) condition[1] = 1;

					//条件3， 端点不能删除
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						sum = sum + abs(b[i]);
					}
					if (sum >= 2) condition[2] = 1;

					//条件4， 孤立点不能删除
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						if (b[i] == 1) sum++;
					}
					if (sum >= 1) condition[3] = 1;

					//条件5， 连通性检测
					if (func_nc8_CPU(b) == 1) condition[4] = 1;

					//条件6，宽度为2的骨架只能删除1边
					sum = 0;
					for (i = 1; i <= 8; i++)
					{
						if (b[i] != -1)
						{
							sum++;
						}
						else
						{
							copy = b[i];
							b[i] = 0;
							if (func_nc8_CPU(b) == 1) sum++;
							b[i] = copy;
						}
					}
					if (sum == 8) condition[5] = 1;

					if (condition[0] && condition[1] && condition[2] && condition[3] && condition[4] && condition[5])
					{
						img[y*w + x] = GRAY; //可以删除，置位GRAY，GRAY是删除标记，但该信息对后面像素的判断有用
						counter++;
					}
				}
			}

			if (counter != 0)
			{
				for (y = 0; y < h; y++)
				{
					for (x = 0; x < w; x++)
					{
						if (img[y*w + x] == GRAY)
							img[y*w + x] = BLACK;

					}
				}
			}

		} while (counter != 0);
		for (int i = 0; i < h; i++)
		{

			for (int j = 0; j < w; j++)
			{
				if (img[j + i * w] == WHITE)
					output_array[j + i * w + z * w * h] = true;
				else
					output_array[j + i * w + z * w * h] = false;
			}
		}
	}
}