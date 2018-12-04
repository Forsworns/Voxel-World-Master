#ifndef VOXELMAP
#define VOXELMAP
#include <stdlib.h> // malloc, calloc, free
#define VX_MALLOC(T, N) ((T*) malloc(N * sizeof(T)))
#define VX_FREE(T) free(T)
#define VX_CALLOC(T, N) ((T*) calloc(N * sizeof(T), 1))		// calloc���ʼ����malloc����

// ��˫����������ͻ���⣬˫����Ϊ�˱����ͷ�
typedef struct vx_hash_table_node {
    struct vx_hash_table_node* next;
    struct vx_hash_table_node* prev;
    void* data;
} vx_hash_table_node_t;

#endif