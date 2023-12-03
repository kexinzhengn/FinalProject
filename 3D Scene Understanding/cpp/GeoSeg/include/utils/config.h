//
// Created by Zhijian QIAO on 2021/4/25.
//

#ifndef SRC_CONFIG_H
#define SRC_CONFIG_H

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <glog/logging.h>
#include <pcl/point_types.h>
#include "tic_toc.h"

namespace config {

    extern std::string project_path, config_file;
    extern Eigen::Vector3f resolution;
	extern int oc_depth;
    extern double plane_ransac_iter, plane_ransac_thd, plane_merge_thd_p, plane_merge_thd_n;

    template<typename T>
    T get(const YAML::Node &node, const std::string &key, const T &default_value) {
        if (!node[key]) {
            LOG(INFO) << "Key " << key << " not found, using default value: " << default_value;
            return default_value;
        }
        T value = node[key].as<T>();
        LOG(INFO) << "Key " << key << " found, using value: " << value;
        return value;
    }

    template<typename T>
    T get(const YAML::Node &node, const std::string &father_key, const std::string &key, const T &default_value) {
        if (!node[father_key] || !node[father_key][key]) {
            LOG(INFO) << "Key " << father_key << "/" << key << " not found, using default value: " << default_value;
            return default_value;
        }
        T value = node[father_key][key].as<T>();
        LOG(INFO) << "Key " << father_key << "/" << key << " found, using value: " << value;
        return value;
    }

    void readParameters(std::string config_file_);
}

void InitGLOG(std::string config_path);

#endif //SRC_CONFIG_H
