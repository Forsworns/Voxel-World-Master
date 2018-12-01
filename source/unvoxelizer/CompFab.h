//
//  CompFab.h
//  voxelizer
//
//
//

#ifndef voxelizer_CompFab_h
#define voxelizer_CompFab_h

#define EPSILON 1e-9
#define USE_DOUBLE false

#include <cmath>

namespace CompFab
{
    #if(USE_DOUBLE)
    typedef double precision_type;
    #else
    typedef float precision_type;
    #endif
    //Data Types
    typedef struct Vec3Struct
    {
        
        Vec3Struct();
        Vec3Struct(precision_type x, precision_type y, precision_type z);

        union
        {
            precision_type m_pos[3];
            struct { precision_type m_x,m_y,m_z; };
        };
        
		//无实际意义仅仅用于map创建，不能逐个比较否则总会形成不严格的偏序关系：a<b，但是c<a,c>b
		inline bool operator < (const Vec3Struct &other) const {
			return m_x * 1777 + m_y * 233 + m_z < other.m_x * 1777 + other.m_y * 233 + other.m_z;
		}
        inline precision_type & operator[](unsigned int index) { return m_pos[index]; }
        inline const precision_type & operator[](unsigned int index) const { return m_pos[index]; }
        inline void operator+=(const Vec3Struct &a)
        {
            m_x += a.m_x;
            m_y += a.m_y;
            m_z += a.m_z;
        }
        
        void normalize();
        
    }Vec3;

    //Data Types
    typedef struct Vec3iStruct
    {
        
        Vec3iStruct();
        Vec3iStruct(precision_type x, precision_type y, precision_type z);
        union
        {
            int m_pos[3];
            struct {int m_x,m_y,m_z;};
        };
        
        inline int & operator[](unsigned int index) { return m_pos[index]; }
        inline const int & operator[](unsigned int index) const { return m_pos[index]; }
        
    }Vec3i;

    //Data Types
    typedef struct Vec2fStruct
    {
        
        Vec2fStruct();
        Vec2fStruct(precision_type x, precision_type y);
        
        union
        {
            float m_pos[2];
            struct { float m_x,m_y; };
        };
        
        inline float & operator[](unsigned int index) { return m_pos[index]; }
        inline const float & operator[](unsigned int index) const { return m_pos[index]; }
        
    }Vec2f;

    
    //NOTE: Ray direction must be normalized
    typedef struct RayStruct
    {
        
        RayStruct();
        RayStruct(Vec3 &origin, Vec3 &direction);
        
        Vec3 m_origin;
        Vec3 m_direction;
        
    } Ray;
    
    typedef struct TriangleStruct
    {
		TriangleStruct() {}
        TriangleStruct(Vec3 &v1, Vec3 &v2,Vec3 &v3);
        
        Vec3 m_v1, m_v2, m_v3;
        
    }Triangle;
    
	//Some useful operations
	CompFab::Vec3i operator+(const Vec3i & v1, const Vec3i & v2);

	//Compute v1 - v2
    Vec3 operator-(const Vec3 &v1, const Vec3 &v2);
    
    Vec3 operator+(const Vec3 &v1, const Vec3 &v2);
    
	Vec3 operator*(const precision_type &a, const Vec3 &v2);

    //Cross Product
    Vec3 operator%(const Vec3 &v1, const Vec3 &v2);
    
    //Dot Product
    precision_type operator*(const Vec3 &v1, const Vec3 &v2);
    
    
    //Grid structure for Voxels，Square voxels only
    typedef struct VoxelGridStruct
    {
        // lowerLeft是单位空间的左下角，单位空间指的是spacing规模的空间，spacing规模也即一单位体素代表的实际长度比例
        VoxelGridStruct(Vec3 lowerLeft, unsigned int dimX, unsigned int dimY, unsigned int dimZ, precision_type spacing);
        ~VoxelGridStruct();

        void save_binvox(const char * filename);

        inline bool & isInside(unsigned int i, unsigned int j, unsigned int k)
        {
            
            return m_insideArray[k*(m_dimX*m_dimY)+j*m_dimY + i];
        }
        
        bool *m_insideArray;
        unsigned int m_dimX, m_dimY, m_dimZ, m_size;
        precision_type m_spacing;
        Vec3 m_lowerLeft;
        
    } VoxelGrid;
}

#endif

