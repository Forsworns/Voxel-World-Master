#include "includes/args.h"
#include "includes/CompFab.h"
#include "includes/Mesh.h"
#include "includes/utils.h"
#include "includes/marchingTable.h"
#include <tclap/CmdLine.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
// #include <cstdlib>
#include <cstdlib>
#include "includes/cuda_math.h"

using namespace std;

enum FileFormat { obj, binvox };

int INF = 10000;

namespace Device{
	struct Point
	{
		float3 pos;
		float3 normal;
		double weight;
		Point() :pos(make_float3(0, 0, 0)), normal(make_float3(0, 0, 0)), weight(0) {}
		Point(float3 p, float3 n, double w) : pos(p), normal(n), weight(w) {}
	};
}

struct VoxelizerArgs : Args {
	// path to files
	std::string input, output;
	FileFormat format;
	bool double_thick;
	// voxelization settings
	int size;
	float precision;
	// width, height, depth;
	int samples;
	// for reconstruction
	int file_type;
	int pre, device, method, func;
};

enum ModelType
{
	BINVOX,
	PCD
};

// construct the command line arguments
VoxelizerArgs * parseArgs(int argc, char *argv[]) {
	VoxelizerArgs * args = new VoxelizerArgs();

	// Define the command line object
	TCLAP::CmdLine cmd("A simple voxelization and unvoxelization utility.", ' ', "0.0");

	TCLAP::UnlabeledValueArg<std::string> input( "input", "path to input (.obj, .binvox, .pcd)", true, "", "string");
	TCLAP::UnlabeledValueArg<std::string> output("output","path to save", true, "", "string");

	TCLAP::ValueArg<std::string> format("q", "format","voxel grid save format - obj|binvox", false, "binvox", "string");
	TCLAP::ValueArg<int> size(  "r","resolution", "voxelization resolution",  false, 32, "int");
	TCLAP::ValueArg<int> precision("e", "precision", "reduces the artifact", false, 2.0f, "float");
	TCLAP::ValueArg<int> samples( "s","samples", "number of sample rays per vertex",  false, -1, "int");

	TCLAP::MultiSwitchArg verbosity( "v", "verbose", "Verbosity level. Multiple flags for more verbosity.");
	TCLAP::SwitchArg double_thick( "o", "double", "Flag for processing double-thick meshes. Uses (num_intersections/2)%2 for occupancy checking.", false);

	TCLAP::ValueArg<int> file_type("t", "type", "file type for reconstruction", false, 0, "int");
	TCLAP::ValueArg<int> pre("p", "prepare","prepare-1, not-0", false, 1, "int");

	TCLAP::ValueArg<int> device("d", "device", "CPU-0,GPU-1", false, 1, "int");
	TCLAP::ValueArg<int> method("m", "method", "method1-1,method2-0", false, 0, "int");
	TCLAP::ValueArg<int> func("f","function","reconstruction-0, voxelization-1", false, 1, "int");

	// Add args to command line object and parse
	cmd.add(input); cmd.add(output);  // order matters for positional args
	cmd.add(size); cmd.add(format); 
	cmd.add(precision);
	// cmd.add(width); cmd.add(height); cmd.add(depth); 
	cmd.add(verbosity); cmd.add(samples); cmd.add(double_thick);
	cmd.add(file_type);
	cmd.add(pre);
	cmd.add(device);
	cmd.add(method);
	cmd.add(func);
	cmd.parse( argc, argv );

	// store in wrapper struct
	args->input  = input.getValue();
	args->output = output.getValue();
	args->size   = size.getValue();
	args->precision = precision.getValue();
	args->samples  = samples.getValue();
	args->verbosity  = verbosity.getValue();
	args->double_thick  = double_thick.getValue();
	args->file_type = file_type.getValue();
	args->pre = pre.getValue();
	args->device = device.getValue();
	args->method = method.getValue();
	args->func = func.getValue();

	args->debug(1) << "input:     " << args->input  << std::endl;
	args->debug(1) << "output:    " << args->output << std::endl;

	char fl = format.getValue().at(0);
	if (fl == 'b' || fl == 'B') {
		args->format = binvox;
		args->debug(1) << "save format: binvox" << std::endl;
	} else if (fl == 'o' || fl == 'O') {
		args->format = obj;
		args->debug(1) << "save format: obj" << std::endl;
	} else {
		args->debug(0) << "Unknown file format specified, use one of: (o) obj, (b) binvox" << std::endl;
	}

	args->debug(1) << "size:      " << args->size   << std::endl;
	args->debug(1) << "precision:     " << args->precision  << std::endl;
	args->debug(1) << "samples:   " << args->samples << std::endl;
	args->debug(1) << "verbosity: " << args->verbosity << std::endl;
	if (args->double_thick) args->debug(1) << "Processing mesh as double-thick." << std::endl;

	return args;
}

typedef std::vector<CompFab::Triangle> TriangleList;

TriangleList g_triangleList;
CompFab::VoxelGrid *g_voxelGrid;

bool loadMesh(const char *filename, unsigned int dim, bool normalize)
{
	g_triangleList.clear();
	
	Mesh *tempMesh = new Mesh(filename, normalize); // 是否归一化
	
	CompFab::Vec3 v1, v2, v3;

	//copy triangles to global list
	for(unsigned int tri =0; tri<tempMesh->t.size(); ++tri)
	{
		v1 = tempMesh->v[tempMesh->t[tri][0]];
		v2 = tempMesh->v[tempMesh->t[tri][1]];
		v3 = tempMesh->v[tempMesh->t[tri][2]];
		g_triangleList.push_back(CompFab::Triangle(v1,v2,v3));
		//std::cout << v1.m_x << v2.m_x << v3.m_x << std::endl;
	}

	//Create Voxel Grid
	CompFab::Vec3 bbMax, bbMin;
	BBox(*tempMesh, bbMin, bbMax);
	std::cout << bbMax[0]<<" " << bbMax[1]<<" " << bbMax[2] <<std::endl;
	//Build Voxel Grid
	double bbX = bbMax[0] - bbMin[0];
	double bbY = bbMax[1] - bbMin[1];
	double bbZ = bbMax[2] - bbMin[2];
	double spacing;
	
	if(bbX > bbY && bbX > bbZ)
	{// 根据最大的一维来确定整个空间的大小，做成正方体
		spacing = bbX/(double)(dim-2);
	} else if(bbY > bbX && bbY > bbZ) {
		spacing = bbY/(double)(dim-2);
	} else {
		spacing = bbZ/(double)(dim-2);
	}
	
	CompFab::Vec3 hspacing(0.5*spacing, 0.5*spacing, 0.5*spacing);

	g_voxelGrid = new CompFab::VoxelGrid(bbMin-hspacing, dim, dim, dim, spacing);

	delete tempMesh;
	return true;
}



void saveVoxelsToObj(const char * outfile)
{
	Mesh box;
	Mesh mout;
	int nx = g_voxelGrid->m_dimX;
	int ny = g_voxelGrid->m_dimY;
	int nz = g_voxelGrid->m_dimZ;
	double spacing = g_voxelGrid->m_spacing;
	
	CompFab::Vec3 hspacing(0.5*spacing, 0.5*spacing, 0.5*spacing);
	
	for (int ii = 0; ii < nx; ii++) {
		for (int jj = 0; jj < ny; jj++) {
			for (int kk = 0; kk < nz; kk++) {
				if(!g_voxelGrid->isInside(ii,jj,kk)){
					continue;
				}
				CompFab::Vec3 coord(0.5f + ((double)ii)*spacing, 0.5f + ((double)jj)*spacing, 0.5f+((double)kk)*spacing);
				CompFab::Vec3 box0 = coord - hspacing;
				CompFab::Vec3 box1 = coord + hspacing;
				makeCube(box, box0, box1);
				mout.append(box);
			}
		}
	}

	mout.save_obj(outfile);
}

bool save(VoxelizerArgs *args) {
	switch (args->format) {
		case obj:
			saveVoxelsToObj((args->output + ".obj").c_str());
			break;
		case binvox:
			g_voxelGrid->save_binvox((args->output + ".binvox").c_str());
			break;
		default:
			args->debug(0) << "Failed to save - no file type specified." << std::endl;
			return false;
	}
	return true;
}

// 判断每个体素在模型内外
extern void kernel_wrapper_1(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick);
extern void useCpuToVoxelize(int samples, int w, int h, int d, CompFab::VoxelGrid *g_voxelGrid, std::vector<CompFab::Triangle> triangles, bool double_thick);

// 借助mesh中的三角形的aabb包围盒构造体素表示
extern void kernel_wrapper_2(int w, int h, int d, float precision, CompFab::VoxelGrid* g_voxelGrid, std::vector<CompFab::Triangle> triangles);

extern void hilditchThin(bool* inside_array, bool* output_array, int w, int h, int d);

extern void prepare(bool useGPU, Device::Point *voxelPointMap, const long long int &blockNum, bool *voxel, bool *voxelThined, const int &dimX, const int &dimY, const int&dimZ, vector<Point> &points, float &originX, float &originY, float &originZ, float scale, float threhold, vector<Triangle> &triangleVector);
extern void kernel_wrapper_3(bool *voxelThinned, Device::Point *voxelPointMap, const int &dimX, const int &dimY, const int&dimZ, float &originX, float &originY, float &originZ, float scale, float threhold, vector<Triangle> &triangleVector);
extern void useCpuToUnvoxelize(bool *voxel, bool *voxelThined, const int &dimX, const int &dimY, const int&dimZ, float &originX, float &originY, float &originZ, float scale, float threhold, vector<Triangle> &triangleVector);
extern void outputToPly(string,vector<Triangle> &);

int main(int argc, char *argv[])
{
	VoxelizerArgs *args = parseArgs(argc, argv);
	const bool useGPU = args->device;
	const bool useWhichMethod = args->method;
	const bool voxelOrUnvoxel = args->func;
	
	if (voxelOrUnvoxel) {
		args->debug(0) << "\nLoading Mesh" << std::endl;
		loadMesh(args->input.c_str(), args->size, false);

		clock_t start = clock();
		if (useGPU) {
			args->debug(0) << "Voxelizing in the GPU, this might take a while." << std::endl;
		}
		else {
			args->debug(0) << "Voxeling in the CPU, this might take a long time" << std::endl;
		}
		
		if (args->samples > -1) args->debug(0) << "Randomly choosing " << args->samples << " directions." << std::endl;

		if (useWhichMethod) {
			if (useGPU) {
				kernel_wrapper_1(args->samples, args->size, args->size, args->size, g_voxelGrid, g_triangleList, args->double_thick);
			}
			else {
				useCpuToVoxelize(args->samples, args->size, args->size, args->size, g_voxelGrid, g_triangleList, args->double_thick);
			}
		}
		else {
			kernel_wrapper_2(args->size, args->size, args->size, args->precision, g_voxelGrid, g_triangleList);
			// 用viewer时出现双层，是因为被当作了实心模型，viewer会自动删除实心只留下表面
		}

		// Summary: teapot.obj (9000 triangles) @ 512x512x512, 3 samples in: 15 seconds
		args->debug(0) << "Summary: "
			<< utils::split(args->input, '/').back()
			<< " (" << g_triangleList.size() << " triangles)"
			<< " @ " << g_voxelGrid->m_dimX << "x" << g_voxelGrid->m_dimY << "x" << g_voxelGrid->m_dimZ;
		if (args->samples > 0)
			args->debug(0) << ", " << args->samples << " samples";
		else args->debug(0) << ", 1 sample";
		args->debug(0) << " in: " << float(clock() - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

		args->debug(0) << "Saving Results." << std::endl;
		if (!save(args)) {
			args->debug(0) << "Failed to save! Exiting." << std::endl;
		};
	}
	else {
		bool isPrepare = args->pre;
		ifstream fin;
		string fileName;
		ModelType modeType = ModelType(args->file_type);
		vector<Point> points;
		float originX, originY, originZ;
		double scale;
		int dimX, dimY, dimZ;
		float threhold = 0;
		vector<Triangle> triangleVector;

		bool *voxel, *voxelThined;
		long long int blockNum;

		switch (modeType)
		{
		case BINVOX:
		{
			char title[12];
			fileName = args->input.c_str();
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
			cout << "正在读入..." << title << endl;


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
			fileName = args->input.c_str();
			dimX = 256;
			dimY = 256;
			dimZ = 256;
			vector<CompFab::Vec3> pointCloud;
			fin.open(fileName);
			char title[80];
			do {
				fin.getline(title, 80);
				cout << title << endl;
			} while (strcmp(title, "DATA ascii") != 0);
			cout << "正在读入..." << endl;
			while (!fin.eof()) {
				float x, y, z;
				fin >> x >> y >> z;
				pointCloud.emplace_back(CompFab::Vec3(x, y, z));
			}

			fin.close();

			// 计算origin(左下角)
			float xMin = INF, yMin = INF, zMin = INF, xMax = -INF, yMax = -INF, zMax = -INF;
			for (vector<CompFab::Vec3>::iterator it = pointCloud.begin(); it < pointCloud.end(); ++it) {
				xMin = xMin < it->m_x ? xMin : it->m_x;
				yMin = yMin < it->m_y ? yMin : it->m_y;
				zMin = zMin < it->m_z ? zMin : it->m_z;
				xMax = xMax > it->m_x ? xMax : it->m_x;
				yMax = yMax > it->m_y ? yMax : it->m_y;
				zMax = zMax > it->m_z ? zMax : it->m_z;
			}
			originX = xMin, originY = yMin, originZ = zMin;

			// 计算scale
			if (xMax - xMin > yMax - yMin && yMax - yMin > zMax - zMin) {
				scale = (xMax - xMin) / (dimX - 1);
			}
			else if (xMax - xMin < yMax - yMin && yMax - yMin > zMax - zMin) {
				scale = (yMax - yMin) / (dimY - 1);
			}
			else {
				scale = (zMax - zMin) / (dimZ - 1);
			}

			// 将点云映射到体素空间得到voxel数组z,y,x
			blockNum = dimX * dimY * dimZ;
			voxel = new bool[blockNum]();
			for (vector<CompFab::Vec3>::iterator it = pointCloud.begin(); it < pointCloud.end(); ++it) {
				int z = floor((it->m_z - originZ) / scale);
				int y = floor((it->m_y - originY) / scale);
				int x = floor((it->m_x - originX) / scale);
				voxel[z*dimX*dimY + y * dimX + x] = true;
			}

			break;
		}
		default:
			return -2;
		}
		// 细化处理
		voxelThined = new bool[blockNum]();
		hilditchThin(voxel, voxelThined, dimX, dimY, dimZ);
		Device::Point *voxelPointMap = new Device::Point[blockNum];
		prepare(useGPU, voxelPointMap, blockNum, voxel, voxelThined, dimX, dimY, dimZ, points, originX, originY, originZ, scale, threhold, triangleVector);
		if (isPrepare) return 0;
		clock_t start = clock();
		if (useGPU) {
			kernel_wrapper_3(voxelThined, voxelPointMap, dimX, dimY, dimZ, originX, originY, originZ, scale, threhold, triangleVector);
		}
		else {
			useCpuToUnvoxelize(voxel, voxelThined, dimX, dimY, dimZ, originX, originY, originZ, scale, threhold, triangleVector);
		}
		args->debug(0) << "unvoxelize in: " << float(clock() - start) / CLOCKS_PER_SEC << " seconds\n正在保存至文件..." << std::endl;
		outputToPly(args->output, triangleVector);
	}
	
	return 0;
}