/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/
#include "dataset/scannet.h"
#include <map>
#include <fstream>
#include <iostream>
#include <vector>

std::map<std::string, std::vector<int>> CLASS_COLOR = {
        {"unannotated", {0, 0, 0}},
        {"floor", {143, 223, 142}},
        {"wall", {171, 198, 230}},
        {"cabinet", {0, 120, 177}},
        {"bed", {255, 188, 126}},
        {"chair", {189, 189, 57}},
        {"sofa", {144, 86, 76}},
        {"table", {255, 152, 153}},
        {"door", {222, 40, 47}},
        {"window", {197, 176, 212}},
        {"bookshelf", {150, 103, 185}},
        {"picture", {200, 156, 149}},
        {"counter", {0, 190, 206}},
        {"desk", {252, 183, 210}},
        {"curtain", {219, 219, 146}},
        {"refridgerator", {255, 127, 43}},
        {"bathtub", {234, 119, 192}},
        {"shower curtain", {150, 218, 228}},
        {"toilet", {0, 160, 55}},
        {"sink", {110, 128, 143}},
        {"otherfurniture", {80, 83, 160}}
};

std::vector<int> SEMANTIC_IDXS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                  11, 12, 14, 16, 24, 28, 33, 34, 36, 39};

std::vector<std::string> SEMANTIC_NAMES = {"wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
                                           "bookshelf", "picture", "counter", "desk", "curtain", "refridgerator",
                                           "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"};

std::vector<int> remapper(150, -100);
void init_remapper() {
    for (int i = 0; i < SEMANTIC_IDXS.size(); ++i) {
        remapper[SEMANTIC_IDXS[i]] = i;
    }
}

pcl::PointCloud<pcl::PointXYZRGBL>::Ptr paintCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
    int N = cloud->size();
    init_remapper();
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGBL>);
    for (int i = 0; i < N; ++i) {
        const auto& point = cloud->points[i];
        pcl::PointXYZRGBL point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        int label = remapper[int(point.intensity)];
        if (label < 0) {
            point_rgb.label = 0;
            point_rgb.r = 0;
            point_rgb.g = 0;
            point_rgb.b = 0;
        } else {
            point_rgb.label = label;
            point_rgb.r = CLASS_COLOR[SEMANTIC_NAMES[label]][0];
            point_rgb.g = CLASS_COLOR[SEMANTIC_NAMES[label]][1];
            point_rgb.b = CLASS_COLOR[SEMANTIC_NAMES[label]][2];
        }
        cloud_rgb->push_back(point_rgb);
    }
    return cloud_rgb;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr GetCloudI(const std::string filename) {
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "error: failed to load point cloud " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points =
            fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    cloud->resize(num_points);

    for (int i = 0; i < num_points; i++) {
        auto &pt = cloud->at(i);
        pt.x = buffer[i * 4];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
        pt.intensity = buffer[i * 4 + 3];
    }

    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr GetCloud(const std::string filename) {
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "error: failed to load point cloud " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points =
            fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->resize(num_points);

    for (int i = 0; i < num_points; i++) {
        auto &pt = cloud->at(i);
        pt.x = buffer[i * 4];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
    }

    return cloud;
}
