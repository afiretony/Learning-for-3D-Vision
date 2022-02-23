# 16-889 Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid
**Optimized voxel grid:**

Note that there's no direct rendering method for voxels in `pytorch3d`, the voxel was first transformed to mesh by `pytorch3d.ops.cubify` and then rendered by mesh render.



![v1](voxelsrc.png)

**Ground Truth voxel:**

![gt](voxeltgt.png)

### 1.2. Fitting a point cloud (10 points)
**Optimized point cloud:**

![pcd](t1.png)

**Ground Truth:**

![gt](t2.png)

### 1.3. Fitting a mesh
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.

**Optimized mesh:**

![mesh](meshsrc.png)

![gt](meshtgt.png)

## 2. Reconstructing 3D from single view

This section involves training a single view to 3D pipeline for voxels, point clouds and meshes.

### 2.1. Image to voxel grid
---
### Decoder:
In order to transfer information of 2D feature maps into 3D volumes. I designed a network structure which is similar to Pix2Vox-F. There are four 3D transposed convolutional layers. In detail, the first three 3D transposed convolutional layers are of a kernel size of $4^3$, with stride of 2 and padding of 1. Each transposed convolutional layer is followed by a batch normalization layer and a ReLU activation except for the last layer followed by a sigmoid activation layer. The numbers of output channels of the transposed convolutional layers are 64, 32, 8, and 1, respectively.

### Results:

From left to right: RGB image input for prediction, predicted voxel, ground truth mesh

<img src="figures/vox/gt_img_0.png" alt="rgb" style="zoom:120%;" /><img src="figures/vox/vox_0.png" alt="pred" style="zoom:50%;" /><img src="figures/vox/gt_mesh_0.png" alt="gt" style="zoom:70%;" />

<img src="figures/vox/gt_img_150.png" alt="rgb" style="zoom:120%;" /><img src="figures/vox/vox_150.png" alt="pred" style="zoom:50%;" /><img src="figures/vox/gt_mesh_150.png" alt="gt" style="zoom:70%;" />

<img src="figures/vox/gt_img_70.png" alt="rgb" style="zoom:120%;" /><img src="figures/vox/vox_70.png" alt="pred" style="zoom:50%;" /><img src="figures/vox/gt_mesh_70.png" alt="gt" style="zoom:70%;" />

### 2.2. Image to point cloud (15 points)

---
### Decoder:

The decoder for generating 3D point cloud I designed contains four 2D transposed convolution layer followed by a fully connected layer that has size $(num\;of\;points)\times3$ and then reshaped to point cloud coordinates. There are batch normalization layers between transposed convolution layers and I used `Tanh` for activation.  

### Results:
<img src="figures/point/gt_img_0.png" alt="rgb" style="zoom:120%;" /><img src="figures/point/point_0.png" alt="pred" style="zoom:70%;" /><img src="figures/point/gt_mesh_0.png" alt="gt" style="zoom:70%;" />



<img src="figures/point/gt_img_10.png" alt="rgb" style="zoom:120%;" /><img src="figures/point/point_10.png" alt="pred" style="zoom:70%;" /><img src="figures/point/gt_mesh_70.png" alt="gt" style="zoom:70%;" />

<img src="figures/point/gt_img_200.png" alt="rgb" style="zoom:120%;" /><img src="figures/point/point_200.png" alt="pred" style="zoom:70%;" /><img src="figures/point/gt_mesh_200.png" alt="gt" style="zoom:70%;" />

### 2.3. Image to mesh (15 points)
---
### Decoder:

The decoder structure I implemented for this network is simply four fully-connected layers followed by `Tanh` activation layer. The number of feature for each layer is `[512, 1024, 2048, 4096]` and outputs $(num\;of\;points)\times3$. The training was made for deforming vertices instead of directly on vertices coordinates.

### Results:

<img src="figures/mesh/gt_img_0.png" alt="rgb" style="zoom:120%;" /><img src="figures/mesh/mesh_0.png" alt="pred" style="zoom:50%;" /><img src="figures/mesh/gt_mesh_0.png" alt="gt" style="zoom:70%;" />



<img src="figures/mesh/gt_img_70.png" alt="rgb" style="zoom:120%;" /><img src="figures/mesh/mesh_70.png" alt="pred" style="zoom:50%;" /><img src="figures/mesh/gt_mesh_70.png" alt="gt" style="zoom:70%;" />

<img src="figures/mesh/gt_img_120.png" alt="rgb" style="zoom:120%;" /><img src="figures/mesh/mesh_120.png" alt="pred" style="zoom:50%;" /><img src="figures/mesh/gt_mesh_120.png" alt="gt" style="zoom:70%;" />



### 2.4. Quantitative comparisions(10 points)

Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
Provide an intutive explaination justifying the comparision.

| Presentation | Loss  | F-score@0.05 |
| :----------: | :---: | :----------: |
|    voxel     | 0.162 |    64.588    |
|  pointcloud  | 0.001 |    91.042    |
|     mesh     | 0.010 |    81.547    |

Generally, we can find that F-score is proportional to the loss during training and F-score of `point cloud` > `mesh` > `voxel` . 



### 2.5. Analyse effects of hyperparms variations (10 points)

Analyse the results, by varying an hyperparameter of your choice.
For example `n_points` or `vox_size` or `w_chamfer` or `initial mesh(ico_sphere)` etc.
Try to be unique and conclusive in your analysis.



### 2.6. Interpret your model (15 points)
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.


## 3. (Extra Credit) Exploring some recent architectures.

### 3.1 Implicit network (10 points)
Implement a implicit decoder that takes in as input 3D locations and outputs the occupancy value.
Some papers for inspiration [[1](https://arxiv.org/abs/2003.04618),[2](https://arxiv.org/abs/1812.03828)]

### 3.2 Parametric network (10 points)
Implement a parametric function that takes in as input sampled 2D points and outputs their respective 3D point.
Some papers for inspiration [[1](https://arxiv.org/abs/1802.05384),[2](https://arxiv.org/abs/1811.10943)]
