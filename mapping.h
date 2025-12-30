/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<double>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();

        // ================Mip_Splat_part===================== version 1
        apply_mip_filter = node["apply_mip_filter"].as<bool>();
        mip_filter_var = node["mip_filter_var"].as<float>();
        apply_3d_filter = node["apply_3d_filter"].as<bool>();
        filter_3d_scale = node["filter_3d_Scale"].as<float>();
        filter_3d_update_freq = node["filter_3d_update_freq"].as<int>();
        // ================Mip_Splat_part===================== version 1

        // ================Probabilistic Gaussian-LIC===================== version 2
        // 不确定性相关参数
        enable_probabilistic = node["enable_probabilistic"].as<bool>(false);
        uncertainty_threshold = node["uncertainty_threshold"].as<float>(0.7f);
        uncertainty_beta = node["uncertainty_beta"].as<float>(3.0f);

        // 几何损失参数
        lambda_depth = node["lambda_depth"].as<float>(0.1f);
        lambda_normal = node["lambda_normal"].as<float>(0.05f);


        // ⬅️ 新增：混合深度损失参数
        lambda_lidar = node["lambda_lidar"].as<float>(1.0f);     // LiDAR 强约束权重
        lambda_sgs = node["lambda_sgs"].as<float>(0.1f);         // SGSNet 软约束权重
        use_scale_alignment = node["use_scale_alignment"].as<bool>(true);

        // 置信度初始化参数
        init_scale_uncertainty_factor = node["init_scale_uncertainty_factor"].as<float>(1.0f);
        init_opacity_uncertainty_factor = node["init_opacity_uncertainty_factor"].as<float>(1.0f);
        // ================Probabilistic Gaussian-LIC===================== version 2


        
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;

     // ================Mip_Splat_part===================== version 1
    bool apply_mip_filter;
    float mip_filter_var;
    bool apply_3d_filter;
    float filter_3d_scale;
    int filter_3d_update_freq;
    // ================Mip_Splat_part===================== version 1

    // ================Probabilistic Gaussian-LIC===================== version 2
    bool enable_probabilistic;
    float uncertainty_threshold;                // 初始化阈值 τ
    float uncertainty_beta;                     // MIP 滤波器不确定性权重 β
    float lambda_depth;                         // 深度损失权重
    float lambda_normal;                        // 法向损失权重
    float init_scale_uncertainty_factor;        // 初始化scale因子
    float init_opacity_uncertainty_factor;      // 初始化opacity因子

    bool use_scale_alignment;
    float lambda_sgs;
    float lambda_lidar;
    // ================Probabilistic Gaussian-LIC===================== version 2

};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    sensor_msgs::ImageConstPtr image_msg;
};