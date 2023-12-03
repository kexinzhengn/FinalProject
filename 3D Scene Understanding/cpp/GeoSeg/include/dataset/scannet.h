/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#ifndef SRC_SCANNET_H
#define SRC_SCANNET_H

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "glog/logging.h"

pcl::PointCloud<pcl::PointXYZRGBL>::Ptr paintCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

pcl::PointCloud<pcl::PointXYZI>::Ptr GetCloudI(const std::string filename);

pcl::PointCloud<pcl::PointXYZ>::Ptr GetCloud(const std::string filename);

#endif //SRC_SCANNET_H
