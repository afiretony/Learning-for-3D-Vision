# 16-889 Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## 0. Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1vbxbSgpBMwqWYhniAyMh590ire8zzeV7/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file [here](https://github.com/learning3d/a2_internal/blob/4d3e93c4e391711559ee9b506cccba412d59ddbd/dataset_location.py#L2)

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

## 1. Exploring loss functions
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)
In this subsection, we will define binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b>.
Define the loss functions [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/losses.py#L5) in `losses.py` file. 
For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.

**Optimized voxel grid:**

Note that there's no direct rendering method for voxels in `pytorch3d`, the voxel was first transformed to mesh by `pytorch3d.ops.cubify` and then rendered by mesh render.



![v1](voxelsrc.png)

**Ground Truth voxel:**

![gt](voxeltgt.png)

### 1.2. Fitting a point cloud (10 points)
In this subsection, we will define chamfer loss that can help us <b> fit a 3D point cloud </b>.
Define the loss functions [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/losses.py#L10) in `losses.py` file.
<b>We expect the you to write your own code for this and not use any pytorch3d utilities. You are allowed to use functions inside pytorch3d.ops.knn such as knn_gather or knn_points</b>

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.

**Optimized point cloud:**

![pcd](t1.png)

**Ground Truth:**

![gt](t2.png)

### 1.3. Fitting a mesh (5 points)
In this subsection, we will define an additional smoothening loss that can help us <b> fit a mesh</b>.
Define the loss functions [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/losses.py#L15) in `losses.py` file.

For this you can use the pre-defined losses in pytorch library.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.

**Optimized mesh:**

![mesh](meshsrc.png)

![gt](meshtgt.png)

## 2. Reconstructing 3D from single view

This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

### 2.1. Image to voxel grid (15 points)
In this subsection, we will define a neural network to decode binary voxel grids.
Define the decoder network [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L21) in `model.py` file, then reference your decoder [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L47) in `model.py` file

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox'`

You need to add the respective visualization code in `eval_model.py`

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.

---
### Decoder:
In order to transfer information of 2D feature maps into 3D volumes. I designed a network structure which is similar to Pix2Vox-F. There are four 3D transposed convolutional layers. In detail, the first three 3D transposed convolutional layers are of a kernel size of $4^3$, with stride of 2 and padding of 1. Each transposed convolutional layer is followed by a batch normalization layer and a ReLU activation except for the last layer followed by a sigmoid activation layer. The numbers of output channels of the transposed convolutional layers are 64, 32, 8, and 1, respectively.

| loss | F-score |
|------|---------|
|      |         |

### Results:
![rgb](/figures/vox/)
![pred](/figures/vox/)
![gt](/figures/vox/)


### 2.2. Image to point cloud (15 points)

In this subsection, we will define a neural network to decode point clouds.
Similar as above, define the decoder network [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L25) in `model.py` file, then reference your decoder [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L52) in `model.py` file

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point'`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.

---
### Decoder:

### Results:
| loss | F-score |
|------|---------|
|      |         |

![rgb](/figures//)
![pred](/figures//)
![gt](/figures//)

### 2.3. Image to mesh (15 points)
In this subsection, we will define a neural network to decode mesh.
Similar as above, define the decoder network [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L31) in `model.py` file, then reference your decoder [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L57) in `model.py` file

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need. We also encourage the student to try different mesh initializations [here](https://github.com/learning3d/a2_internal/blob/037b19c92e7f2ed41edd4e4667c66360998a7f32/model.py#L28)


After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh'`

You need to add the respective visualization code in `eval_model.py`.

On your webpage, you should include visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.

---
### Decoder:

### Results:

![rgb](/figures//)
![pred](/figures//)
![gt](/figures//)

### 2.4. Quantitative comparisions(10 points)
Quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids.
Provide an intutive explaination justifying the comparision.

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point`


On your webpage, you should include the average test F1 score at 0.05 threshold for voxelgrid, pointcloud and the mesh network.

| Representation |  Loss | F-score |
|----------------|-------|---------|
|      voxel     | 0.162 | 64.588  |
|   pointcloud   | 0.001 | 91.042  |
|      mesh      | 0.010 | 81.547  |


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
