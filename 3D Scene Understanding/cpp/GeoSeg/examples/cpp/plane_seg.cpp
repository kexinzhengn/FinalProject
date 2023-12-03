#include <iostream>
#include <string>
#include "utils/file_manager.h"
#include "global_definition/global_definition.h"
#include "dataset/scannet.h"
#include "segmentation/octree_seg.h"
#include "utils/tic_toc.h"
#include "utils/config.h"
#include "segmentation/voxel_clustering.h"
#include <pcl/filters/voxel_grid.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/features/normal_3d_omp.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <pcl/surface/gp3.h>

using namespace std;

int main(int argc, char **argv) {

    std::string config_path = point_sam::WORK_SPACE_PATH + "/configs/scannet.yaml";
    InitGLOG(config_path);
    config::readParameters(config_path);
    TicToc ticToc;
    // traverse the dataset_path
	std::string pcd_file = point_sam::WORK_SPACE_PATH + "/data/ScanNet/scene0011_01.pcd";
	// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = GetCloud(bin_file);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
	PCL_ERROR("Couldn't read file your_point_cloud.pcd\n");
	return (-1);
    }

	ticToc.tic();
	OctreeSeg oc_seg(cloud);
	oc_seg.segmentation();
	LOG(INFO) << "segmentation time: " << ticToc.toc() << " ms";
	pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_ocs = oc_seg.getLabeledRGBCloud();
	// pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud = ground;
	pcl::PointCloud<pcl::PointXYZRGBL>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZRGBL>);
	*combined_cloud = *cloud_ocs; // Copy the first point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr oc_seg_ground_cloud = oc_seg.ground_cloud;


	// pcl::io::savePCDFileASCII("/home/uav/graphGen/data/ground.pcd", *oc_seg_ground_cloud);
	// pcl::io::savePCDFileASCII("/home/uav/graphGen/data/oc_seg.pcd", *cloud_ocs);
	visualizeCloud<pcl::PointXYZRGBL>(combined_cloud, pcd_file);

    return 0;
}
