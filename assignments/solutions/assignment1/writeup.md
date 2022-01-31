# 16-889 Assignment 1: Rendering Basics with PyTorch3D

**Chenhao Yang**

Goals: In this assignment, you will learn the basics of rendering with PyTorch3D, explore 3D representations, and practice constructing simple geometry. You may find it also helpful to follow the [Pytorch3D tutorials](https://github.com/facebookresearch/pytorch3d).



## 1. Practicing with Cameras

### 1.1. 360-degree Renders (5 points)

![360 view of a row](output/360.gif)

### 1.2 Re-creating the Dolly Zoom (10 points)

The [Dolly Zoom](https://en.wikipedia.org/wiki/Dolly_zoom) is a famous camera effect, first used in the Alfred Hitchcock film [Vertigo](https://www.youtube.com/watch?v=G7YJkBcRWB8). The core idea is to change the focal length of the camera while moving the camera in a way such that the subject is the same size in the frame, producing a rather unsettling effect.

![dolly zoom](output/dolly.gif)

## 2. Practicing with Meshes

### 2.1 Constructing a Tetrahedron (5 points)

In this part, you will practice working with the geometry of 3D meshes. Construct a tetrahedron mesh and then render it from multiple viewpoints.  Your tetrahedron does not need to be a regular tetrahedron (i.e. not all faces need to be equilateral triangles) as long as it is obvious from the renderings that the shape is a tetrahedron.

You will need to manually define the vertices and faces of the mesh. Once you have the vertices and faces, you can define a single-color texture, similarly to the cow in `render_cow.py`. Remember that the faces are the vertex indices of the triangle mesh.

It may help to draw a picture of your tetrahedron and label the vertices and assign 3D coordinates.

**On your webpage, show a 360-degree gif animation of your tetrahedron. Also, list how many vertices and (triangle) faces your mesh should have.**

![cube](output/q2_1.gif)

There should be four vertices and four faces.

```python
vertices = torch.tensor([[math.sqrt(3),0,-1],[0,0,2],[-math.sqrt(3),0,-1],[0,3,0]])
faces = torch.tensor([[1,0,3],[3,2,1],[0,2,3],[0,1,2]])	
```



### 2.2 Constructing a Cube (5 points)

Construct a cube mesh and then render it from multiple viewpoints. Remember that we are still working with triangle meshes, so you will need to use two sets of triangle faces to represent one face of the cube.

**On your webpage, show a 360-degree gif animation of your cube. Also, list how many vertices and (triangle) faces your mesh should have.**

![cube](output/q2_2.gif)

There should be 8 vertices and 12 faces.

```python
vertices = torch.tensor([[1.,0,1.],[1.,0.,-1.],[-1,0,-1],[-1,0,1],
  [1,2,1],[1,2,-1],[-1.,2.,-1.],[-1.,2.,1.]])

faces = torch.tensor([[1,0,2],[0,3,2],
                      [2,3,6],[7,6,3],
                      [1,4,0],[1,5,4],
                      [1,2,6],[1,6,5],
                      [0,4,7],[0,7,3],
                      [4,5,7],[4,6,7],
                      ])
```



## 3. Re-texturing a mesh (10 points)

Now let's practice re-texturing a mesh. For this task, we will be retexturing the cow mesh such that the color smoothly changes from the front of the cow to the back of the cow.

**In your submission, describe your choice of `color1` and `color2`, and include a gif of the rendered mesh.**

![cow](output/q3.gif)

color of my choice:

```python
color1 = torch.tensor([0,0,1]) # blue
color2 = torch.tensor([1,1,1]) # white
```



## 4. Camera Transformations (20 points)

When working with 3D, finding a reasonable camera pose is often the first step to producing a useful visualization, and an important first step toward debugging.

Running `python -m starter.camera_transforms` produces the following image using the camera extrinsics rotation `R_0` and translation `T_0`:

![Cow render](images/transform_none.jpg)

What are the relative camera transformations that would produce each of the following output images? You shoud find a set (R_relative, T_relative) such that the new camera extrinsics with `R = R_relative @ R_0` and `T = R_relative @ T_0 + T_relative` produces each of the following images:

![Cow render](images/transform1.jpg) ![Cow render](images/transform3.jpg) ![Cow render](images/transform4.jpg) ![Cow render](images/transform2.jpg)

**In your report, describe in words what R_relative and T_relative should be doing and include the rendering produced by your choice of R_relative and T_relative.**

 ```Python
 R_relative=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]
 ```

![cam1](output/cam1.jpg)



```python
T_relative=[0, 0, 1]
```

![cam2](output/cam2.jpg)

```python
T_relative=[0.5, 0, 0]
```

![cam3](output/cam3.jpg)

```python
rotate_angle = -90. / 180. * math.pi
R_relative=[math.cos(rotate_angle), 0, -math.sin(rotate_angle)], [0, 1, 0], [math.sin(rotate_angle), 0, math.cos(rotate_angle)]
T_relative=[-3., 0, 3]
```

![cam4](output/cam4.jpg)

## 5. Rendering Generic 3D Representations

The simplest possible 3D representation is simply a collection of 3D points, each possibly associated with a color feature. PyTorch3D provides functionality for rendering point clouds.

Similar to the mesh rendering, we will need a `PointCloud` object consisting of 3D points and colors, a camera from which to view the point cloud, and a Pytorch3D Point  Renderer which we have wrapped similarly to the Mesh Renderer.

To construct a point cloud, use the `PointCloud` class:

```python
points = ...  # 1 x N x 3
rgb = ...  # 1 x N x 3
point_cloud = pytorch3d.structures.PointCloud(
    points=points, features=rgb
)
```

As with all the mesh rendering, everything should be batched.

The point renderer takes in a point cloud and a camera and returns a B x H x W x 4 rendering, similar to the mesh renderer.

```
from starter.utils import get_points_renderer
points_renderer = get_points_renderer(
    image_size=256,
    radius=0.01,
)
rend = points_renderer(point_cloud, cameras=cameras)
image = rend[0, ..., :3].numpy()  # (B, H, W, 4) -> (H, W, 3).
```

To see a full working example of rendering a point cloud, see `render_bridge` in `starter/render_generic.py`.

If you run `python -m starter.render_generic --render point_cloud`, you should get the following output:

![bridge](images/bridge.jpg)

### 5.1 Rendering Point Clouds from RGB-D Images (10 points)

In this part, we will practice rendering point clouds constructed from 2 RGB-D images from the [Common Objects in 3D Dataset](https://github.com/facebookresearch/co3d).

![plant](images/plant.jpg)

In `render_generic.py`, the `load_rgbd_data` function will load the data for 2 images of the same plant. The dictionary should contain the RGB image, a depth map, a mask, and a Pytorch3D camera corresponding to the pose that the image was taken from.

You should use the `unproject_depth_image` function in `utils.py` to convert a depth image into a point cloud (parameterized as a set of 3D coordinates and corresponding color values). The `unproject_depth_image` function uses the camera intrinsics and extrinisics to cast a ray from every pixel in the image into world  coordinates space. The ray's final distance is the depth value at that pixel, and the color of each point can be determined from the corresponding image pixel.

Construct 3 different point clouds:

1. The point cloud corresponding to the first image
2. The point cloud corresponding to the second image
3. The point cloud formed by the union of the first 2 point clouds.

Try visualizing each of the point clouds from various camera viewpoints. We suggest starting with cameras initialized 6 units from the origin with equally spaced azimuth values.

**In your submission, include a gif of each of these point clouds side-by-side.**

### 5.2 Parametric Functions (10 points)

A parametric function generates a 3D point for each point in the source domain. For example, given an elevation `theta` and azimuth `phi`, we can parameterize the surface of a unit sphere as `(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi))`.

By sampling values of `theta` and `phi`, we can generate a sphere point cloud. You can render a sphere point cloud by calling `python -m starter.render_generic --render parametric`. Note that the amount of samples can have an effect on the appearance quality. Below, we show the output with a 100x100 grid of (phi, theta) pairs (`--num_samples 100`) as well as a  1000x1000 grid (`--num_samples 1000`). The latter may take a long time to run on CPU.

![Sphere 100](images/sphere_100.jpg) ![Sphere 1000](images/sphere_1000.jpg)

Your task is to render a [torus](https://en.wikipedia.org/wiki/Torus) point cloud by sampling its parametric function.

**In your writeup, include a 360-degree gif of your torus point cloud, and make sure the hole is visible. You may choose to texture your point cloud however you wish.**

### 5.3 Implicit Surfaces (15 points)

In this part, we will explore representing geometry as a function in the form of an implicit function. In general, given a function F(x, y, z), we can define the surface to be the zero level-set of F i.e. (x,y,z) such that F(x, y, z) = 0. The function F can be a mathematical equation or even a neural network. To visualize such a representation, we can discretize the 3D space and evaluate the implicit function, storing the values in a voxel grid. Finally, to recover the mesh, we can run the [marching cubes](https://en.wikipedia.org/wiki/Marching_cubes) algorithm to extract the 0-level set.

In practice, we can generate our voxel coordinates using `torch.meshgrid` which we will use to query our function (in this case mathematical ones). Once we have our voxel grid, we can use the  [`mcubes`](https://github.com/pmneila/PyMCubes) library convert into a mesh.

A sample sphere mesh can be constructed implicitly and rendered by calling `python -m starter.render_generic --render implicit`. The output should like like this:

![Sphere mesh](images/sphere_mesh.jpg)

Your task is to render a torus again, this time as a mesh defined by an implicit function.

**In your writeup, include a 360-degree gif of your torus mesh, and make sure the hole is visible. In addition, discuss some of the tradeoffs between rendering as a mesh vs a point cloud. Things to consider might include rendering speed, rendering quality, ease of use, memory usage, etc.**

## 6. Do Something Fun (10 points)

Now that you have learned to work with various 3D represenations and render them, it is time to try something fun. Create your own 3D structures, or render something in an interesting way, or creatively texture, or anything else that appeals to you - the (3D) world is your oyster! If you wish to download additional meshes, [Free3D](https://free3d.com/) is a good place to start.

**Include a creative use of the tools in this assignment on your webpage!**

## (Extra Credit) 7. Sampling Points on Meshes (10 points)

We will explore how to obtain point clouds from triangle meshes. One obvious way to do this is to simply discard the face information and treat the vertices as a point cloud. However, this might be unresonable if the faces are not of equal size.

Instead, as we saw in the lectures, a solution to this problem is to use a uniform sampling of the surface using stratified sampling. The procedure is as follows:

1. Sample a face with probability proportional to the area of the face
2. Sample a random [barycentric coordinate](https://en.wikipedia.org/wiki/Barycentric_coordinate_system) uniformly
3. Compute the corresponding point using baricentric coordinates on the selected face.

For this part, write a function that takes a triangle mesh and the number of samples and outputs a point cloud. Then, using the cow mesh, randomly sample 10, 100, 1000, and 10000 points. **Render each pointcloud and the original cow mesh side-by-side, and include the gif in your writeup.\***