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

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <chrono>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "mapping.h"
#include "camera.h"
#include "eigen_utils.h"
#include "general_utils.h"
#include "optim_utils.h"
#include "tinyply.h"
#include "geometry_head.h"

#include "simple-knn/spatial.h"
#include "rasterizer/renderer.h"

const double C0 = 0.28209479177387814;
inline double RGB2SH(double color) {return (color - 0.5) / C0;}
inline torch::Tensor RGB2SH(torch::Tensor& rgb) {return (rgb - 0.5f) / C0;}

class Dataset
{
public:
    Dataset(const Params& prm)
      : fx_(prm.fx), fy_(prm.fy), cx_(prm.cx), cy_(prm.cy),
        select_every_k_frame_(prm.select_every_k_frame),
        all_frame_num_(0), is_keyframe_current_(false) {}
        
    void addFrame(Frame& cur_frame);

public:
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    int select_every_k_frame_;


    int all_frame_num_;
    bool is_keyframe_current_;

    Eigen::aligned_vector<Eigen::Matrix3d> R_wc_;
    Eigen::aligned_vector<Eigen::Vector3d> t_wc_;

    Eigen::aligned_vector<Eigen::Vector3d> pointcloud_;
    Eigen::aligned_vector<Eigen::Vector3d> pointcolor_;
    std::vector<float> pointdepth_;
    
    std::vector<std::shared_ptr<Camera>> train_cameras_;
    std::vector<std::shared_ptr<Camera>> test_cameras_;

    // ================Probabilistic Gaussian-LIC===================== version 2
    torch::Tensor current_dense_depth_;   // 当前关键帧对应的稠密深度图（CUDA Tensor）
    torch::Tensor current_uncertainty_map_; // 当前关键帧对应的不确定性


    // 存储每个关键帧的深度先验和不确定性（用于几何约束损失）
    std::unordered_map<int, torch::Tensor> prior_depths_;      // keyframe_idx -> depth (H, W)
    std::unordered_map<int, torch::Tensor> uncertainty_maps_;  // keyframe_idx -> uncertainty (H, W)
    std::unordered_map<int, torch::Tensor> sparse_depths_;      // ⬅️ 新增：原始 LiDAR 深度

    // 存储深度先验的辅助函数
    void storePriorDepth(int keyframe_idx, const torch::Tensor& depth, const torch::Tensor& uncertainty,const torch::Tensor& sparse_depth) {
        // 存储到 CPU 以节省 GPU 显存
        prior_depths_[keyframe_idx] = depth.clone().to(torch::kCPU);
        uncertainty_maps_[keyframe_idx] = uncertainty.clone().to(torch::kCPU);
        sparse_depths_[keyframe_idx] = sparse_depth.clone().to(torch::kCPU);  // ⬅️ 新增
    }

    // 检查是否有深度先验
    bool hasPriorDepth(int keyframe_idx) const {
        return prior_depths_.find(keyframe_idx) != prior_depths_.end();
}
    // ⬅️ 新增方法
    bool hasSparseDepth(int keyframe_idx) const {
        return sparse_depths_.find(keyframe_idx) != sparse_depths_.end();
    }

    // ================Probabilistic Gaussian-LIC===================== version 2
};


#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};          \
    this->Tensor_vec_exposure_ = {this->exposure_};

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->exposure_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    GAUSSIAN_MODEL_TENSORS_TO_VEC

class GaussianModel
{
public:
    GaussianModel(const Params& prm);

    torch::Tensor getScaling();
    torch::Tensor getRotation();
    torch::Tensor getXYZ();
    torch::Tensor getFeaturesDc();
    torch::Tensor getFeaturesRest();
    torch::Tensor getOpacity();
    torch::Tensor getCovariance(int scaling_modifier);

    torch::Tensor getExposure();

    void initialize(const std::shared_ptr<Dataset>& dataset);
    void initialize(const std::shared_ptr<Dataset>& dataset, 
                const GeometryOutput* geo_output);  // ← 新增重载
    void saveMap(const std::string& result_path);

    void trainingSetup();

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_uncertainties  // 新增参数
    );
    
    // ================Mip_Splat_part===================== version 1
    void computeMaxSamplingRate(const std::vector<std::shared_ptr<Camera>>& cameras);
    torch::Tensor getRegularizedScaling();
    bool is3DFilterEnable();
    // ================Mip_Splat_part===================== version 1


    // ================Probabilistic Gaussian-LIC===================== version 2
    // 不确定性相关方法
    torch::Tensor getUncertainty();
    void setUncertainty(const torch::Tensor& uncertainty);

    // 概率初始化
    void probabilisticInitialize(
        const std::shared_ptr<Dataset>& dataset,
        const torch::Tensor& dense_depth,
        const torch::Tensor& uncertainty_map
    );

    // 获取不确定性调制后的缩放
    torch::Tensor getUncertaintyModulatedScaling();
    // ================Probabilistic Gaussian-LIC===================== version 2   

public:
    int sh_degree_;
    bool white_background_;
    bool random_background_;
    bool convert_SHs_python_;
    bool compute_cov3D_python_;
    double lambda_erank_;
    double scaling_scale_;

    double position_lr_;
    double feature_lr_;
    double opacity_lr_;
    double scaling_lr_;
    double rotation_lr_;
    double lambda_dssim_;

    bool apply_exposure_;
    double exposure_lr_;
    int skybox_points_num_;
    int skybox_radius_;


    torch::Tensor xyz_;
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    
    torch::Tensor exposure_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_,
                               Tensor_vec_exposure_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    std::shared_ptr<SparseGaussianAdam> sparse_optimizer_;  // 这行代码巨傻逼，SparseGaussianAdam 共有了torch::optim::Optimizer，所以可以指针调用 Optimizer

    std::shared_ptr<torch::optim::Adam> exposure_optimizer_;

    bool is_init_;

    torch::Tensor bg_;

    std::chrono::steady_clock::time_point t_start_;
    std::chrono::steady_clock::time_point t_end_;
    double t_forward_;
    double t_backward_;
    double t_step_;
    double t_optlist_;
    double t_tocuda_;

    // ================Mip_Splat_part===================== version 1
    bool  apply_mip_filter_;
    float  mip_filter_var_;
    bool  apply_3d_filter_;
    float  filter_3d_scale_;
    int  filter_3d_update_freq_;
    torch::Tensor max_sampling_rate_;
    // ================Mip_Splat_part===================== version 1

    // ================Probabilistic Gaussian-LIC===================== version 2
    // 每个高斯的不确定性属性
    torch::Tensor uncertainty_;     // (N ,) 每个高斯的不确定性

    // 概率参数
    bool enable_probabilistic_ = false;
    float uncertainty_threshold_ = 0.7f;          // 初始化阈值 τ
    float uncertainty_beta_ = 3.0f;               // MIP 滤波器不确定性权重 β
    float lambda_depth_ = 0.1f;                   // 深度损失权重
    float lambda_normal_ = 0.05f;                  // 法向损失权重

    float lambda_lidar_ = 1.0f;   // ⬅️ 新增
    float lambda_sgs_ = 0.1f;     // ⬅️ 新增

    float init_scale_uncertainty_factor_ = 1.0f;      // 尺度不确定性因子
    float init_opacity_uncertainty_factor_ = 1.0f;    // 不透明度不确定性因子
    // ================Probabilistic Gaussian-LIC===================== version 2
};

void extend(const std::shared_ptr<Dataset>& dataset, 
            std::shared_ptr<GaussianModel>& pc,
            const GeometryOutput* geo_output = nullptr);  // 可选参数

double optimize(const std::shared_ptr<Dataset>& dataset, 
                std::shared_ptr<GaussianModel>& pc,
                const GeometryOutput* geo_output = nullptr);  // 可选参数
void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, 
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path);
