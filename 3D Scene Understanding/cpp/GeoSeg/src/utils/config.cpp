//
// Created by Zhijian QIAO on 2021/12/24.
//

#include "utils/config.h"
#include <glog/logging.h>
#include "global_definition/global_definition.h"
#include "utils/file_manager.h"
#include <filesystem>

namespace config {

    // benchmark config
    std::string project_path = point_sam::WORK_SPACE_PATH, config_file;
    Eigen::Vector3f resolution = Eigen::Vector3f(1.0, 1.0, 1.0);
    int oc_depth = 3;
    double plane_ransac_iter = 2, plane_ransac_thd = 0.1, plane_merge_thd_p = 0.1, plane_merge_thd_n = 5.0;

    std::vector<double> get_vec(const YAML::Node &node, const std::string &father_key, const std::string &key){
        std::vector<double> vec;
        if (!node[father_key] || !node[father_key][key]) {
            LOG(INFO) << "Key " << father_key << "/" << key << " not found, using default value: ";
            return vec;
        }
        for (const auto &item : node[father_key][key]) {
            vec.push_back(item.as<double>());
        }
        LOG(INFO) << "Key " << father_key << "/" << key << " found, using value: " << vec.size();
        return vec;
    }

    void readParameters(std::string config_file_) {
        config_file = config_file_;
        LOG(INFO) << "config_path: " << config_file;
        std::ifstream fin(config_file);
        if (!fin) {
            std::cout << "config_file: " << config_file << " not found." << std::endl;
            return;
        }

        YAML::Node config_node = YAML::LoadFile(config_file);
        std::vector<double> resolution_vec = get_vec(config_node, "dyna_octree", "resolution");
        if (resolution_vec.size() == 3) {
            resolution = Eigen::Vector3f(resolution_vec[0], resolution_vec[1], resolution_vec[2]);
        }
        oc_depth = get(config_node, "dyna_octree", "depth", oc_depth);
        plane_ransac_iter = get(config_node, "plane_seg", "ransac_iter", plane_ransac_iter);
        plane_ransac_thd = get(config_node, "plane_seg", "ransac_thd", plane_ransac_thd);
        plane_merge_thd_p = get(config_node, "plane_seg", "merge_thd_p", plane_merge_thd_p);
        plane_merge_thd_n = get(config_node, "plane_seg", "merge_thd_n", plane_merge_thd_n);
        plane_merge_thd_n = cos(plane_merge_thd_n / 180.0 * M_PI);
    }
}

void InitGLOG(std::string config_path){

    std::string config_name = config_path.substr(config_path.find_last_of('/') + 1, config_path.find_last_of('.') - config_path.find_last_of('/') - 1);
    std::string log_dir = point_sam::WORK_SPACE_PATH + "/Log/";
    FileManager::CreateDirectory(log_dir);
    log_dir = log_dir + config_name;
    FileManager::CreateDirectory(log_dir);

    google::InitGoogleLogging("");

    std::string filename = std::filesystem::path(config_path).stem().string();
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::string log_file = log_dir + "/" + filename + ".log.";
    google::SetLogDestination(google::INFO, log_file.c_str());

    FLAGS_log_dir = log_dir;
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_log_prefix = false;
    FLAGS_logbufsecs = 0;
//    FLAGS_timestamp_in_logfile_name = true;
}