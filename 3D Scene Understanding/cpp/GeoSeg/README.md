## Point Segment Anything (point-SAM)
### A simple and fast point cloud segmentation algorithm
把点云用体素地图来管理，然后用体素地图来做聚类，这样就可以做到快速的点云分割。如果体素相邻，那么就是同一个聚类。

### Quick Start
```bash
# transform npy to bin file so that we can use it in c++
python examples/python/render_semantic.py
# instance segmentation using point-SAM
# clustering in every semantic class
# save the instance segmentation result in data/kimera10_concat/txt files, zero means unlabelled
# semantic_class = instance_id / 1000
./build/kimera_sam
```