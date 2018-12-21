# Voxel-World-Master
this is a toy voxelizer and unvoxelizer project for CS337 CG 

- Learn from [kctess5's](https://github.com/kctess5/voxelizer) voxelizer at the beginning, and this appilcation is based on his framework. 

- Rewirte another voxelizing algorithm using cuda.

- Implement marching cube algorithm in a easy, special but effective way to unvoxelize .binvox into .ply, which supports .pcd to .ply at the same time. Also accelerate it.

  

---

**original .obj** 

![](.\image\bone_obj.png)

**.obj -> .binvox**

![](./image/v/bone2.png)

**.binvox -> .ply**

![](./image/unv/bone256.png)