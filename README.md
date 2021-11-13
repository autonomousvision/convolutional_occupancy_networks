# Convolutional Occupancy Networks
[**Paper**](https://arxiv.org/pdf/2003.04618.pdf) | [**Supplementary**](http://www.cvlibs.net/publications/Peng2020ECCV_supplementary.pdf) | [**Video**](https://www.youtube.com/watch?v=EmauovgrDSM) | [**Teaser Video**](https://youtu.be/k0monzIcjUo) | [**Project Page**](https://pengsongyou.github.io/conv_onet) | [**Blog Post**](https://autonomousvision.github.io/convolutional-occupancy-networks/) <br>

<div style="text-align: center">
<img src="media/teaser_matterport.gif" width="600"/>
</div>

This repository contains the implementation of the paper:

Convolutional Occupancy Networks  
[Songyou Peng](https://pengsongyou.github.io/), [Michael Niemeyer](https://m-niemeyer.github.io/), [Lars Mescheder](https://is.tuebingen.mpg.de/person/lmescheder), [Marc Pollefeys](https://www.inf.ethz.ch/personal/pomarc/) and [Andreas Geiger](http://www.cvlibs.net/)  
**ECCV 2020 (spotlight)**  

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{Peng2020ECCV,
 author =  {Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, Andreas Geiger},
 title = {Convolutional Occupancy Networks},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2020}}
```
Contact [Songyou Peng](mailto:songyou.pp@gmail.com) for questions, comments and reporting bugs.


## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `conv_onet` using
```
conda env create -f environment.yaml
conda activate conv_onet
```
**Note**: you might need to install **torch-scatter** mannually following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140):
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
```

## Demo
First, run the script to get the demo data:
```
bash scripts/download_demo_data.sh
```
### Reconstruct Large-Scale Matterport3D Scene
You can now quickly test our code on the real-world scene shown in the teaser. To this end, simply run:
```
python generate.py configs/pointcloud_crop/demo_matterport.yaml
```
This script should create a folder `out/demo_matterport/generation` where the output meshes and input point cloud are stored.

**Note**: This experiment corresponds to our **fully convolutional model**, which we train only on the small crops from our synthetic room dataset. This model can be directly applied to large-scale real-world scenes with real units and generate meshes in a sliding-window manner, as shown in the [teaser](media/teaser_matterport.gif). More details can be found in section 6 of our [supplementary material](http://www.cvlibs.net/publications/Peng2020ECCV_supplementary.pdf). For training, you can use the script `pointcloud_crop/room_grid64.yaml`.


### Reconstruct Synthetic Indoor Scene
<div style="text-align: center">
<img src="media/demo_syn_room.gif" width="600"/>
</div>

You can also test on our synthetic room dataset by running: 
```
python generate.py configs/pointcloud/demo_syn_room.yaml
```
## Dataset

To evaluate a pretrained model or train a new model from scratch, you have to obtain the respective dataset.
In this paper, we consider 4 different datasets:

### ShapeNet
You can download the dataset (73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.

### Synthetic Indoor Scene Dataset
For scene-level reconstruction, we create a synthetic dataset of 5000
scenes with multiple objects from ShapeNet (chair, sofa, lamp, cabinet, table). There are also ground planes and randomly sampled walls.

You can download our preprocessed data (144 GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/synthetic_room_dataset` folder.  
**Note**: We also provide **point-wise semantic labels** in the dataset, which might be useful.


Alternatively, you can also preprocess the dataset yourself.
To this end, you can:
* download the ShapeNet dataset as described above.
* check `scripts/dataset_synthetic_room/build_dataset.py`, modify the path and run the code.

### Matterport3D
Download Matterport3D dataset from [the official website](https://niessner.github.io/Matterport/). And then, use `scripts/dataset_matterport/build_dataset.py` to preprocess one of your favorite scenes. Put the processed data into `data/Matterport3D_processed` folder.

### ScanNet
Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet).
Then, you can preprocess data with:
`scripts/dataset_scannet/build_dataset.py` and put into `data/ScanNet` folder.  
**Note**: Currently, the preprocess script normalizes ScanNet data to a unit cube for the comparison shown in the paper, but you can easily adapt the code to produce data with real-world metric. You can then use our fully convolutional model to run evaluation in a sliding-window manner.

## Usage
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our pre-trained models and train new models from scratch.

### Mesh Generation
To generate meshes using a trained model, use
```
python generate.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.

**Use a pre-trained model**  
The easiest way is to use a pre-trained model. You can do this by using one of the config files under the `pretrained` folders.

For example, for 3D reconstruction from noisy point cloud with our 3-plane model on the synthetic room dataset, you can simply run:
```
python generate.py configs/pointcloud/pretrained/room_3plane.yaml
```
The script will automatically download the pretrained model and run the generation. You can find the outputs in the `out/.../generation_pretrained` folders

Note that the config files are only for generation, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pretrained model.


We provide the following pretrained models:
```
pointcloud/shapenet_1plane.pt
pointcloud/shapenet_3plane.pt
pointcloud/shapenet_grid32.pt
pointcloud/shapenet_3plane_partial.pt
pointcloud/shapenet_pointconv.pt
pointcloud/room_1plane.pt
pointcloud/room_3plane.pt
pointcloud/room_grid32.pt
pointcloud/room_grid64.pt
pointcloud/room_combine.pt
pointcloud/room_pointconv.pt
pointcloud_crop/room_grid64.pt
voxel/voxel_shapenet_1plane.pt
voxel/voxel_shapenet_3plane.pt
voxel/voxel_shapenet_grid32.pt
```

### Evaluation
For evaluation of the models, we provide the script `eval_meshes.py`. You can run it using:
```
python eval_meshes.py CONFIG.yaml
```
The script takes the meshes generated in the previous step and evaluates them using a standardized protocol. The output will be written to `.pkl/.csv` files in the corresponding generation folder which can be processed using [pandas](https://pandas.pydata.org/).

**Note:** We follow previous works to use "use 1/10 times the maximal edge length of the current object’s bounding box as unit 1" (see [Section 4 - Metrics](http://www.cvlibs.net/publications/Mescheder2019CVPR.pdf)). In practice, this means that we multiply the Chamfer-L1 by a factor of 10 for reporting the numbers in the paper.

### Training
Finally, to train a new network from scratch, run:
```
python train.py CONFIG.yaml
```
For available training options, please take a look at `configs/default.yaml`.

## Further Information
Please also check out the following concurrent works that either tackle similar problems or share similar ideas:
- [[CVPR 2020] Jiang et al. - Local Implicit Grid Representations for 3D Scenes](https://arxiv.org/abs/2003.08981)
- [[CVPR 2020] Chibane et al. Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion](https://arxiv.org/abs/2003.01456)
- [[ECCV 2020] Chabra et al. - Deep Local Shapes: Learning Local SDF Priors for Detailed 3D Reconstruction](https://arxiv.org/abs/2003.10983)
