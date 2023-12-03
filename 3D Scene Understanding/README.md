
## ScanNet data structure
Run traditional dense volumetric fusion using Open3d on ScanNet dataset.Please make sure that you download the [ScanNet.zip](https://drive.google.com/drive/folders/1QBIrZ4vKTLhvT_LdQ4cXZZPdd0LTzFpk) and unzip and put it in the directory as ```./data2/ScanNet```. 

Construct the dataset folder as follows,
```bash
ScanNet
|---splits
    |---val.txt
|---scans
    |---scene0011_01
        |---color
        |---depth
        |---pose
        |---intrinsic
        |---prediction_no_augment(generate from RAM-Grounded-SAM)
    |---...
|---output
```
The ```split``` folder defines the list of scans. 

Run the dense mapping module as follows,

```bash
python scripts/dense_mapping.py 
```
And you will get ```mesh_o3d_256.ply``` in the "./data2/ScanNet/scans/scene0011_01/"
## Implementation of geometric plane segmentation(just tried in the project)

```bash
cd ./cpp/GeoSeg
mkdir build
cd build
cmake ..
make
```
After compilation, run
```bash
./plane_seg
```
The final segmentation result is produced by semantic mapping module, this is just an exploration for the implementation of geometric plane segmentation.

## Run Python version of instance mapping node 
The python verision is based on existed interface of Open3D. Please install opencv-python, numpy, open3d, scipy, scikit-learn. conda environment is recommended.
Run the mapping node,
```bash
python scripts/semantic_mapping.py
```
The output files should be at ```${SCANNET_ROOT}/output/demo```.
If you want to visualize the process, add ```--visualize``` flag.
```bash
python scripts/semantic_mapping.py --visualize
```
To refine the instances volume,
```bash
python scripts/postfuse.py
```
The output files should be at ```${SCANNET_ROOT}/output/demo_refined```. Please make sure that the ```debug``` folder and ```mesh_o3d_256.ply``` is generated before running the refinement module. If you want to produce the contents in the folder ```prediction_no_augment``` on your own, then you need to delete the contents in the ```prediction_no_augment```first, and please follow the instructions in the following ```Grounded-SAM module``` section.

## Grounded-SAM module
The grounded-SAM module is modified from [RAM-Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), to produce the prediction of semantic segmentation of the images, please
```bash
cd ./Grounded-SAM
```
and follow the instructions in the [README.md] in the ```Grounded-SAM``` folder.


