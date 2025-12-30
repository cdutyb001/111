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

#include "gaussian.h"
#include "tensor_utils.h"
#include "loss_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include "geometry_head.h"

#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <limits>
#include <torch/script.h>
#include <memory>

namespace fs = std::filesystem;

void Dataset::addFrame(Frame& cur_frame)
{
    /// image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);  // 0-255
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);  // 0-1

    /// pose
    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);

    /// point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    for (const auto& pt : cloud->points)
    {
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
        Eigen::Vector3d t_cw = - R_cw * t_wc;
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(pt_c(2)));
    }

    /// train & test
    int width = image_rgb.cols, height = image_rgb.rows;
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
        
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".jpg";

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".jpg";

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;  // sh_degree = 3
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)

    is_init_ = false;

    t_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;

    // ================Mip_Splat_part===================== version 1
    apply_mip_filter_ = prm.apply_mip_filter;
    mip_filter_var_ = prm.mip_filter_var;
    apply_3d_filter_ = prm.apply_3d_filter;
    filter_3d_scale_ = prm.filter_3d_scale;
    filter_3d_update_freq_ = prm.filter_3d_update_freq;
    // ================Mip_Splat_part===================== version 1

    // ================Probabilistic Gaussian-LIC===================== version 2
    enable_probabilistic_ = prm.enable_probabilistic;
    uncertainty_threshold_ = prm.uncertainty_threshold;
    uncertainty_beta_ = prm.uncertainty_beta;
    lambda_depth_ = prm.lambda_depth;
    lambda_normal_ = prm.lambda_normal;
    init_scale_uncertainty_factor_ = prm.init_scale_uncertainty_factor;
    init_opacity_uncertainty_factor_ = prm.init_opacity_uncertainty_factor;
    lambda_lidar_ = prm.lambda_lidar;   // ⬅️ 新增
    lambda_sgs_ = prm.lambda_sgs;       // ⬅️ 新增
    // ================Probabilistic Gaussian-LIC===================== version 2
}

bool GaussianModel::is3DFilterEnable()
{
    return GaussianModel::apply_3d_filter_;
}
torch::Tensor GaussianModel::getScaling()
{
    return torch::exp(scaling_);
}

torch::Tensor GaussianModel::getRotation()
{
    return torch::nn::functional::normalize(rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return xyz_;
}

torch::Tensor GaussianModel::getFeaturesDc()
{
    return features_dc_;
}

torch::Tensor GaussianModel::getFeaturesRest() 
{
    return features_rest_;
}

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getCovariance(int scaling_modifier)  // 获取协方差
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset)
/*
    这段代码的核心功能是将通过 SFM（如 Colmap）或 SLAM 系统获得的稀疏点云（Sparse Point Cloud），
    转换为 3DGS 优化所需的初始高斯分布参数，并将其加载到 GPU 上以备梯度下降优化。
*/
{
    /// foreground   前景高斯初始化
    // step1 这一部分将稀疏点云转换为初始的各向异性高斯球。
    int num = static_cast<int>(dataset->pointcloud_.size());
    assert(num > 0);
    torch::Tensor fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();  // (n, 3)
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);  // 16
    // 分配外观特征（Features）张量
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    // 分配尺度（Scales）张量
    torch::Tensor scales = torch::zeros({num}, torch::kFloat32).cuda();


    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num; ++i) 
    {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];

        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();

        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());

        double d = dataset->pointdepth_[i];
        scales.index({i}) = std::log(scaling_scale_ * d / f);    // *****关键点**** 使用了基于投影几何的启发方式，具有一致性的像素覆盖范围，在大场景中具有比k-nn更强的深度适应性。
    }
    scales = scales.unsqueeze(1).repeat({1, 3});                // (n, 3)    让每个点在xyz上都有相同的初始深度，便于后续的3dgs分布建模

    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;              //  初始化为单位四元数 $q=[1,0,0,0]$，即初始高斯没有旋转，轴向与世界坐标系对齐。 // torch::indexing::Slice()表示选取所有行
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1) 激活函数域
    /// sky initialization 
    /*
        这段代码相当于生成一个不透明度很高、密度很大的使用高斯椭球拼成的蓝色背景板
    */
    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }

    this->xyz_ = fused_point_cloud.requires_grad_();  // (n, 3)
    // this->xyz_ = fused_point_cloud.requires_grad_(false);  // fix xyz
    this->features_dc_ = features.index({torch::indexing::Slice(),     //取第 0 个系数（0阶球谐），即 Diffuse 颜色。
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();  // (n, 1, 3)
    this->features_rest_ = features.index({torch::indexing::Slice(),   //取第 1 到 15 个系数（1~3阶球谐），即 View-dependent 细节。
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();  // (n, 15, 3)
    this->scaling_ = scales.requires_grad_();  // (n, 3)
    this->rotation_ = rots.requires_grad_();  // (n, 4)
    this->opacity_ = opacities.requires_grad_();  // (n, 1)

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();  // (3, 4)
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC  
    // 这是一个宏（Macro），很可能用于将上述分散的 xyz_, features_, scaling_ 等张量收集到一个 torch::nn::ParameterDict 或者一个列表中，方便传递给 PyTorch 的优化器（如 Adam）进行统一管理。
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;37m Init Map with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS" 
              << ",\033[0m";

    // ================Mip_Splat_part===================== version 1
    // 用于初始化3D Filter,初始化最大采样率tensor

    max_sampling_rate_ = torch::ones({fused_point_cloud.size(0)},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // 打印Mip Filter配置

    if(apply_mip_filter_)
    {
        std::cout << std::fixed << std::setprecision(2)
                  << "\033[1;35m Mip Filter enabled (var=" << mip_filter_var_ << ")\033[0m";
    }
    if (apply_3d_filter_)
    {
        std::cout << std::fixed << std::setprecision(2)
                  << "\033[1;35m 3D Filter enabled (s=" << filter_3d_scale_ << ")\033[0m";
    }
    // ================Mip_Splat_part===================== version 1

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset, 
                               const GeometryOutput* geo_output)
/*
    重载版本：支持基于稠密深度图的概率初始化
    结合白皮书中的 UWGI (Uncertainty-Weighted Gaussian Initialization) 策略
*/
{
    // 如果没有 geo_output 或未启用概率模式，回退到原始初始化
    if (!enable_probabilistic_ || geo_output == nullptr || 
        geo_output->dense_depth.numel() == 0) {
        initialize(dataset);  // 调用原版本
        return;
    }

    torch::NoGradGuard no_grad;
    
    // ========== 双路径融合初始化 ==========
    // 路径1：原始LiDAR稀疏点云（高置信度）
    // 路径2：SGSNet稠密深度补全（带不确定性）
    
    // ---------- 路径1：处理原始稀疏点云 ----------
    int num_sparse = static_cast<int>(dataset->pointcloud_.size());
    assert(num_sparse > 0);
    
    torch::Tensor sparse_point_cloud = torch::zeros({num_sparse, 3}, torch::kFloat32).cuda();
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    torch::Tensor sparse_features = torch::zeros({num_sparse, 3, deg_2}, torch::kFloat32).cuda();
    torch::Tensor sparse_scales = torch::zeros({num_sparse}, torch::kFloat32).cuda();
    torch::Tensor sparse_uncertainties = torch::zeros({num_sparse}, torch::kFloat32).cuda();  // LiDAR点不确定性=0
    
    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num_sparse; ++i) {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];
        
        sparse_point_cloud.index({i, 0}) = pt_w.x();
        sparse_point_cloud.index({i, 1}) = pt_w.y();
        sparse_point_cloud.index({i, 2}) = pt_w.z();
        
        sparse_features.index({i, 0, 0}) = RGB2SH(color.x());
        sparse_features.index({i, 1, 0}) = RGB2SH(color.y());
        sparse_features.index({i, 2, 0}) = RGB2SH(color.z());
        
        double d = dataset->pointdepth_[i];
        sparse_scales.index({i}) = std::log(scaling_scale_ * d / f);
    }
    
    // ---------- 路径2：从稠密深度图采样补充点 ----------
    auto dense_depth = geo_output->dense_depth.to(torch::kCUDA);  // (H, W)
    auto uncertainty_map = geo_output->uncertainty_map.to(torch::kCUDA);  // (H, W)
    
    float fx = static_cast<float>(dataset->fx_);
    float fy = static_cast<float>(dataset->fy_);
    float cx = static_cast<float>(dataset->cx_);
    float cy = static_cast<float>(dataset->cy_);
    float focal = (fx + fy) / 2.0f;
    
    int H = dense_depth.size(0);
    int W = dense_depth.size(1);
    
    // 获取最新相机位姿
    Eigen::Matrix3d R_wc = dataset->R_wc_.back();
    Eigen::Vector3d t_wc = dataset->t_wc_.back();
    
    // 创建稀疏点掩码（避免重复采样）
    cv::Mat sparse_mask = cv::Mat::zeros(H, W, CV_8UC1);
    {
        auto sparse_pc_cpu = sparse_point_cloud.to(torch::kCPU);
        auto sparse_accessor = sparse_pc_cpu.accessor<float, 2>();
        Eigen::Matrix3d R_cw = R_wc.transpose();
        Eigen::Vector3d t_cw = -R_cw * t_wc;
        
        for (int i = 0; i < num_sparse; ++i) {
            Eigen::Vector3d pt_w(sparse_accessor[i][0], sparse_accessor[i][1], sparse_accessor[i][2]);
            Eigen::Vector3d pt_c = R_cw * pt_w + t_cw;
            if (pt_c.z() > 0) {
                int u = static_cast<int>(fx * pt_c.x() / pt_c.z() + cx);
                int v = static_cast<int>(fy * pt_c.y() / pt_c.z() + cy);
                if (u >= 0 && u < W && v >= 0 && v < H) {
                    // 在稀疏点周围创建排除区域（5x5）
                    for (int dv = -2; dv <= 2; ++dv) {
                        for (int du = -2; du <= 2; ++du) {
                            int nu = u + du, nv = v + dv;
                            if (nu >= 0 && nu < W && nv >= 0 && nv < H) {
                                sparse_mask.at<uchar>(nv, nu) = 255;
                            }
                        }
                    }
                }
            }
        }
    }


    
    
    // 从稠密深度图中采样补充点（跳过稀疏点区域）
    auto depth_cpu = dense_depth.to(torch::kCPU);
    auto uncertainty_cpu = uncertainty_map.to(torch::kCPU);
    auto depth_accessor = depth_cpu.accessor<float, 2>();
    auto uncertainty_accessor = uncertainty_cpu.accessor<float, 2>();
    
    std::vector<Eigen::Vector3d> dense_points;
    std::vector<float> dense_depths;
    std::vector<float> dense_uncerts;
    
    int sample_step = 4;  // 降采样因子    原本是4，为了增加高斯点数我改成了2，增加了4倍
    for (int v = 0; v < H; v += sample_step) {
        for (int u = 0; u < W; u += sample_step) {
            // 跳过已有稀疏点的区域
            if (sparse_mask.at<uchar>(v, u) > 0) continue;
            
            float depth = depth_accessor[v][u];
            float uncertainty = uncertainty_accessor[v][u];
            
            // UWGI 置信度门控：只初始化不确定性低于阈值的点
            if (depth <= 0.1f || uncertainty > uncertainty_threshold_) continue;
            
            // 反投影到相机坐标系
            float x_c = (u - cx) * depth / fx;
            float y_c = (v - cy) * depth / fy;
            float z_c = depth;
            
            // 转换到世界坐标系
            Eigen::Vector3d p_c(x_c, y_c, z_c);
            Eigen::Vector3d p_w = R_wc * p_c + t_wc;
            
            dense_points.push_back(p_w);
            dense_depths.push_back(depth);
            dense_uncerts.push_back(uncertainty);
        }
    }
    
    int num_dense = static_cast<int>(dense_points.size());
    
    // ---------- 合并两路径点云并应用UWGI ----------
    int total_num = num_sparse + num_dense;
    
    torch::Tensor fused_point_cloud = torch::zeros({total_num, 3}, torch::kFloat32).cuda();
    torch::Tensor features = torch::zeros({total_num, 3, deg_2}, torch::kFloat32).cuda();
    torch::Tensor scales = torch::zeros({total_num, 3}, torch::kFloat32).cuda();
    torch::Tensor rots = torch::zeros({total_num, 4}, torch::kFloat32).cuda();
    torch::Tensor opacities = torch::zeros({total_num, 1}, torch::kFloat32).cuda();
    torch::Tensor uncertainties = torch::zeros({total_num}, torch::kFloat32).cuda();
    
    // 填充稀疏点（前 num_sparse 个）
    fused_point_cloud.index({torch::indexing::Slice(0, num_sparse), torch::indexing::Slice()}) = sparse_point_cloud;
    features.index({torch::indexing::Slice(0, num_sparse), torch::indexing::Slice(), torch::indexing::Slice()}) = sparse_features;
    
    // 稀疏点：各向同性缩放，完全置信
    torch::Tensor sparse_scales_3d = sparse_scales.unsqueeze(1).repeat({1, 3});
    scales.index({torch::indexing::Slice(0, num_sparse), torch::indexing::Slice()}) = sparse_scales_3d;
    opacities.index({torch::indexing::Slice(0, num_sparse), torch::indexing::Slice()}) = 
        general_utils::inverse_sigmoid(0.1f * torch::ones({num_sparse, 1}, torch::kFloat32).cuda());
    // uncertainties 前 num_sparse 个保持为 0（已初始化）
    
    // 填充稠密补充点（后 num_dense 个）
    for (int i = 0; i < num_dense; ++i) {
        int idx = num_sparse + i;
        
        // 位置
        fused_point_cloud.index({idx, 0}) = dense_points[i].x();
        fused_point_cloud.index({idx, 1}) = dense_points[i].y();
        fused_point_cloud.index({idx, 2}) = dense_points[i].z();
        
        // 颜色（使用灰色占位，后续由光度优化校正）
        features.index({idx, 0, 0}) = RGB2SH(0.5);
        features.index({idx, 1, 0}) = RGB2SH(0.5);
        features.index({idx, 2, 0}) = RGB2SH(0.5);
        
        float uncertainty = dense_uncerts[i];
        float depth = dense_depths[i];
        
        // ===== UWGI 公式实现 =====
        // 基础尺度
        float base_scale = std::log(scaling_scale_ * depth / focal);
        
        // 1. 各向异性缩放：深度方向拉长
        // s_xy = s_base, s_z = s_base * (1 + β * σ)
        float scale_factor = 1.0f + uncertainty_beta_ * uncertainty;
        scales.index({idx, 0}) = base_scale;  // x
        scales.index({idx, 1}) = base_scale;  // y  
        scales.index({idx, 2}) = base_scale * scale_factor;  // z（深度方向）
        
        // 2. 不透明度调制：不确定的点更透明
        // α_init = α_base * (1 - σ * factor)
        float base_opacity = 0.1f * (1.0f - uncertainty * init_opacity_uncertainty_factor_);
        base_opacity = std::max(0.01f, std::min(0.99f, base_opacity));
        opacities.index({idx, 0}) = std::log(base_opacity / (1.0f - base_opacity));
        
        // 3. 存储不确定性
        uncertainties.index({idx}) = uncertainty;
    }
    
    // 旋转初始化为单位四元数
    rots.index({torch::indexing::Slice(), 0}) = 1;
    
    // ---------- 天空盒初始化（与原版相同） ----------
    if (skybox_points_num_ > 0) {
        int sky_num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({sky_num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({sky_num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({sky_num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);
        
        torch::Tensor sky_features = torch::zeros({sky_num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;
        
        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2)).unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({sky_num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({sky_num, 1}, torch::kFloat32).cuda());
        torch::Tensor sky_uncertainties = torch::zeros({sky_num}, torch::kFloat32).cuda();
        
        // 天空盒放在最前面
        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
        uncertainties = torch::cat({sky_uncertainties, uncertainties}, 0);
    }
    
    // ---------- 设置模型参数 ----------
    this->xyz_ = fused_point_cloud.requires_grad_();
    this->features_dc_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();
    this->features_rest_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();
    this->scaling_ = scales.requires_grad_();
    this->rotation_ = rots.requires_grad_();
    this->opacity_ = opacities.requires_grad_();
    this->uncertainty_ = uncertainties;  // 不需要梯度
    
    if (apply_exposure_) {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();
    }
    
    GAUSSIAN_MODEL_TENSORS_TO_VEC
    
    // Mip Filter 初始化
    max_sampling_rate_ = torch::ones({fused_point_cloud.size(0)}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 打印初始化统计
    std::cout << std::fixed << std::setprecision(2)
              << "\033[1;37m Init Map with " << double(fused_point_cloud.size(0)) / 10000 << "w GS"
              << " (sparse: " << num_sparse << ", dense: " << num_dense << ")"
              << ",\033[0m";
    
    if (enable_probabilistic_) {
        float mean_unc = uncertainties.index({torch::indexing::Slice(skybox_points_num_)}).mean().item<float>();
        std::cout << "\033[1;35m [UWGI] τ=" << uncertainty_threshold_ 
                  << ", β=" << uncertainty_beta_
                  << ", mean_σ=" << mean_unc << "\033[0m";
    }
    
    if (apply_mip_filter_) {
        std::cout << "\033[1;35m Mip Filter enabled (var=" << mip_filter_var_ << ")\033[0m";
    }
    if (apply_3d_filter_) {
        std::cout << "\033[1;35m 3D Filter enabled (s=" << filter_3d_scale_ << ")\033[0m";
    }
    
    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::saveMap(const std::string& result_path)
{

    //=================根据时间的保存路径========================
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);
    std::ostringstream time_ss;
    time_ss<<std::setfill('0')
            <<std::setw(2)<<(now_tm->tm_mon+1)<<"-"
            <<std::setw(2)<<now_tm->tm_mday<<"-"
            <<std::setw(2)<<now_tm->tm_hour<<"-"
            <<std::setw(2)<<now_tm->tm_min;

    std::string timestamp_dir = time_ss.str();
    std::string timestamp_path = result_path+"/"+timestamp_dir;
    fs::create_directories(timestamp_path);
    //=================根据时间的保存路径========================
    std::string pc_path = timestamp_path + "/point_cloud.ply";

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu(); // 断开梯度计算图
    // torch::Tensor normals = torch::zeros_like(xyz);
    //是球谐函数的DC分量（0阶），原始形状 [N, 1, 3]（RGB三通道）
    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu(); //.contiguous() 
    // 是球谐函数的高阶分量，形状 [N, (deg+1)²-1, 3]
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    // 不透明度，形状 [N, 1]
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // 3D缩放（对数空间），形状 [N, 3]
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // rotation_：四元数旋转，形状 [N, 4]
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    // 打开二进制文件流，使用 filebuf + ostream 组合创建二进制输出流。
    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file; //创建 PLY 文件对象并添加属性  ，使用 tinyply 库来处理PLY格式。

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // // normals
    // result_file.add_properties_to_element(
    //     "vertex", {"nx", "ny", "nz"},
    //     tinyply::Type::FLOAT32, normals.size(0),
    //     reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
    //     tinyply::Type::INVALID, 0);

    // f_dc 添加 DC 球谐系数       动态生成属性名：f_dc_0, f_dc_1, f_dc_2（对应RGB三通道的DC值）。
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest   添加高阶球谐系数
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModel::trainingSetup()   // 头文件没定义，现在才重载
{
    //初始化主优化器，并确立“位置（XYZ）”为第 0 号参数组
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    // 球谐系数（SH）的分频优化策略
    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);
    /*
    这行代码是一串连续的内存访问和函数调用，我们可以从左向右拆解：
    sparse_optimizer_->param_groups()：
    返回优化器内部维护的所有参数组列表的引用（通常是 std::vector<ParamGroup>&）。
    [1]：
    索引访问。由于上一行代码刚刚将 Tensor_vec_feature_dc_ 追加进去，它是列表中的第二个元素，所以下标是 1。（下标 0 是上一段代码中的 XYZ 位置参数）。
    .options()：
    访问该参数组的配置选项。在 LibTorch 中，每个参数组都有自己独立的一套超参数（Options），这允许不同层或不同类型的参数使用不同的训练策略。
    这行代码显式地告诉优化器：“对于第 1 组参数（即颜色 DC 分量），不要使用默认学习率，请使用我指定的 feature_lr_。”
    .set_lr(feature_lr_)：
    设置器（Setter）。将该组的 学习率（Learning Rate） 修改为变量 feature_lr_ 的值。

    Line 1: “施工队，请把这堆‘颜色数据（DC）’纳入你的管理范围。”（此时，颜色数据被标记为 Group 1）。
    Line 2: “听好了，对于 Group 1（颜色数据），你们干活的速度（学习率）要设定为 feature_lr_，这和刚才设定的 Group 0（位置数据）的速度是不一样的。”
    */

    // 几何形态参数的独立控制
    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    if (apply_exposure_) // 如果开启了曝光补偿（通常用于处理如 Phototourism 等由不同相机、不同曝光时间拍摄的数据集），则初始化一个独立的优化器
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }
}

void GaussianModel::densificationPostfix(    //自适应密度控制（Adaptive Density Control） 中至关重要的一环，即 “后处理（Postfix）” 阶段。
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_uncertainties)  // 新增参数
{
    // 1 数据准备
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = 
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    // 2 获取优化器底层权限
    auto& param_groups = this->sparse_optimizer_->param_groups();  // 获取优化器当前的参数组列表的引用。我们需要修改里面的 params 指针。
    auto& optimizer_state = this->sparse_optimizer_->get_state();  //它是一个 Map，Key 是参数 Tensor 的唯一标识符（Pointer/ID），Value 是其对应的优化状态（包含 $m_t$ 一阶动量, $v_t$ 二阶矩, step 等）。

     //3 核心循环：逐个属性组更新
    for (int group_idx = 0; group_idx < 6; ++group_idx) 
    {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];   //param：这是旧的、长度为 $N$ 的 Tensor，存储着增殖前的所有高斯参数。

        // 4 获取旧 Tensor 的“身份证”
        auto old_param_impl = param.unsafeGetTensorImpl();
        /*
        unsafeGetTensorImpl()：这是一个非常底层的 LibTorch API。它返回 Tensor 在 C++ 层面的实现指针（c10::TensorImpl*）。
        作用：在优化器的 optimizer_state 字典中，这个指针就是查找动量数据的 Key。
        如果不保存这个旧指针，一旦下面执行 cat 操作生成新 Tensor，我们就再也找不到旧参数对应的动量数据了
        */

        // 5 拼接（Concatenation）与梯度重置
        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
        //在维度 0 上拼接。旧参数（长度N） + 新参数（长度M） = 新总参数（长度N+M）。cat会申请一块新的缓存，param只想这块新的缓存
        // if (group_idx == 0) param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_(false);  // fix xyz
        // else param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();  // fix xyz
        group.params()[0] = param;  // 我们将优化器参数组中存储的指针，更新为指向这个新的、更长的 Tensor。从此之后优化器将对这个新的Tensor进行求导。

        // 6 获取新的 Tensor 的" 身份证 " 
        auto new_param_impl = param.unsafeGetTensorImpl(); //获取新生成的 (N+M) 长度 Tensor 的指针。这将成为优化器状态字典中的 新 Key。


        // ****重要****  7 优化器状态迁移（State Migration）
        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end()) 
        {
            //stored_state：取出了旧的 Adam 状态包（包含 exp_avg (m) 和 exp_avg_sq (v)）
            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else  //8. 处理未初始化的情况（Edge Case） 。这种情况通常只在第一次迭代前，或者某些参数还没开始更新时发生。为新参数创建一个空的状态记录
        {
            State new_state;
            new_state.step = 0;
            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;

            optimizer_state[new_param_impl] = new_state;
        }

        // * 更新类成员变量
        optimizable_tensors[group_idx] = param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    // ================Mip_Splat_part===================== 
    //  同步更新max_sampling_rate_,为新增的高斯设置默认采样率。
    // ✅ 新增：同步更新 uncertainty_
    if (enable_probabilistic_ && new_uncertainties.numel() > 0) {
        if (uncertainty_.numel() == 0) {
            uncertainty_ = new_uncertainties;
        } else {
            uncertainty_ = torch::cat({uncertainty_, new_uncertainties}, 0);
        }
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    if(apply_3d_filter_ && max_sampling_rate_.numel()>0)
    {
        int num_new = new_xyz.size(0);
        // 为新增的高斯分配默认采样率（使用1.0，表示需要在下次更新时计算）
        torch::Tensor new_sampling_rates = torch::ones({num_new},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        max_sampling_rate_ = torch::cat({max_sampling_rate_,new_sampling_rates},0);
    }
    // ================Mip_Splat_part===================== 
}

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc,const GeometryOutput* geo_output)
//这个函数的作用是：基于当前的相机观测（通常是关键帧的深度图或稀疏点云），向现有的高斯模型中添加新的高斯点。为了防止点云无限膨胀，它引入了严格的遮挡剔除和冗余检测机制。
{
    // 1 上下文与渲染检测
    // 这是一个 RAII（资源获取即初始化）守卫。在它的作用域内，PyTorch 的自动求导机制被关闭。因为 extend 只是添加数据，不需要计算梯度，关闭它可以大幅节省显存和计算资源。
    torch::NoGradGuard no_grad; 

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda(); // torch::ones 创建全1张量的函数.{3} 表示纬度为1 【1.0，1.0，1.0】
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<1>(render_pkg).squeeze(0);

    // 2. 数据转换：从 C++ Vector 到 GPU Tensor
    int n = dataset->pointcloud_.size();
    std::vector<float> float_point(n * 3);
    std::vector<float> float_color(n * 3);
    for (int i = 0; i < n; ++i) 
    {
        float_point[3 * i + 0] = static_cast<float>(dataset->pointcloud_[i][0]);
        float_point[3 * i + 1] = static_cast<float>(dataset->pointcloud_[i][1]);
        float_point[3 * i + 2] = static_cast<float>(dataset->pointcloud_[i][2]);

        float_color[3 * i + 0] = static_cast<float>(dataset->pointcolor_[i][0]);
        float_color[3 * i + 1] = static_cast<float>(dataset->pointcolor_[i][1]);
        float_color[3 * i + 2] = static_cast<float>(dataset->pointcolor_[i][2]);
    }
    torch::Tensor points = torch::from_blob(float_point.data(), {n, 3}).to(torch::kFloat32).cuda(); 
    //  torch::from_blob LibTorch 核心语法。它不会进行内存拷贝，而是直接根据指针（float_point.data()）创建一个 Tensor 视图（View）
    torch::Tensor colors = torch::from_blob(float_color.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor depths_in_rsp_frame = torch::from_blob(dataset->pointdepth_.data(), {n}).to(torch::kFloat32).cuda();

    // 3. 几何变换：世界系 -> 相机系 -> 图像平面
    /// filter
    auto R_wc = dataset->R_wc_.back();
    auto t_wc = dataset->t_wc_.back();
    auto R_cw = R_wc.transpose();
    auto t_cw = - R_cw * t_wc;
    std::vector<float> float_R_cw(3 * 3);//【0,0,0,0,0,0,0,0,0】
    std::vector<float> float_t_cw(3);
    for (int i = 0; i < 3; ++i)
    {
        float_R_cw[3 * i + 0] = static_cast<float>(R_cw(i, 0));
        float_R_cw[3 * i + 1] = static_cast<float>(R_cw(i, 1));
        float_R_cw[3 * i + 2] = static_cast<float>(R_cw(i, 2));
        float_t_cw[i] = static_cast<float>(t_cw[i]);
    }
    torch::Tensor R_cw_tensor = torch::from_blob(float_R_cw.data(), {3, 3}).to(torch::kFloat32).cuda();
    torch::Tensor t_cw_tensor = torch::from_blob(float_t_cw.data(), {3, 1}).to(torch::kFloat32).cuda();
    //----- 世界系→相机系坐标变换
    auto points_camera = torch::matmul(points, R_cw_tensor.t()) + t_cw_tensor.view({1, 3});  // (n, 3)
    /*
    R_cw_tensor.t()​ - 矩阵转置
    torch::matmul(points, R_cw_tensor.t())​ - 矩阵乘法
    t_cw_tensor.view({1, 3})​ - 形状重塑为行向量  [[1, 2, 3]]  // 形状: [1, 3]。存在广播机制，会将所有行都
    */
    auto depths = points_camera.index({torch::indexing::Slice(), 2});  // (n)
    float fx = static_cast<float>(viewpoint_cam->fx_);
    float fy = static_cast<float>(viewpoint_cam->fy_);
    float cx = static_cast<float>(viewpoint_cam->cx_);
    float cy = static_cast<float>(viewpoint_cam->cy_);
    float focal = (fx + fy) / 2.0;
    // ------ 透视投影（相机系→图像系）
    // 标准的**针孔相机模型（Pinhole Camera Model）**投影公式 u =  fx（X/Z）+cx
    torch::Tensor x_pixel = (points_camera.index({torch::indexing::Slice(), 0}) * fx) / depths + cx; // points_camera.index({torch::indexing::Slice(), 0}) 表示的是所有点的x坐标。也即是所有行的第一个元素*fx
    torch::Tensor y_pixel = (points_camera.index({torch::indexing::Slice(), 1}) * fy) / depths + cy; // torch::indexing::Slice()：相当于 Python 中的 [:]，表示选取该维度所有元素。
    auto pixels = torch::stack({x_pixel, y_pixel}, 1);  // (n, 2)
    pixels = pixels.floor().to(torch::kInt32);

    auto pixels_float = pixels.to(torch::kFloat32);
    // 深度缓冲（Z-Buffer）过滤 —— CPU 端处理
    auto pixels_with_depth = torch::cat({pixels_float, depths.unsqueeze(1)}, 1).to(torch::kCPU);
    auto pixels_depth_a = pixels_with_depth.accessor<float, 2>();
    /*
    accessor<float, 2>()：LibTorch 性能关键点。
    直接操作 Tensor 的开销很大（每次下标访问都会进行动态类型检查和越界检查）。
    Accessor 提供了一个轻量级的、类似数组的接口来访问 Tensor 数据，极其适合在 CPU 循环中使用。
    pixels_depth_a【i】【j】. 允许像二维数一样访问pixels_with_depth的数据。222
    逻辑：
    构建一个 pixel_depth_map（哈希表）。
    遍历所有投影点。如果同一个像素坐标 (x, y) 有多个点落入，只保留**深度最小（最近）**的那一个。
    目的：避免在一个像素上重复生成多个重叠的高斯球，这是最基础的遮挡处理
    */

    // 基于cpu的 z- buffer 去重  // 这一步在 CPU 上进行，目的是处理 “同一个像素对应了多个 3D 点” 的情况。
    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    for (int i = 0; i < pixels_with_depth.size(0); ++i) {
        int x = static_cast<int>(pixels_depth_a[i][0]); // 这里就用到了上面 accessor的二维数组化接口
        int y = static_cast<int>(pixels_depth_a[i][1]);
        float depth = pixels_depth_a[i][2];
        
        // 核心逻辑：如果这个位置还没人，或者当前点比原来的点更近（depth 更小）
        std::string key = std::to_string(x) + "_" + std::to_string(y);  //Key：将 (x, y) 坐标拼成字符串作为哈希表的键。
        if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
            pixel_depth_map[key] = {i, depth};
        }
    }

    // 应用筛选结果
    std::vector<int64_t> keep_indices;   //indices 索引
    for (const auto& item : pixel_depth_map) {
        keep_indices.push_back(item.second.first);    // 将筛选出来的点放到 keep_indices 之中
    }

    //这行代码是一个非常经典的 LibTorch 数据传输操作，分为两步走
    auto keep_indices_tensor = torch::from_blob(  //torch::from_blob(...)：零拷贝视图创建
        keep_indices.data(), 
        {static_cast<int64_t>(keep_indices.size())}, 
        torch::kInt64
    ).to(points.device());  // .to(points.device())：硬件迁移（H2D Copy） ，在没有使用to的时候， keep_indices_tensor仅仅是cpu的一个视图，和keep_indices 共享内存

    auto filtered_points = points.index_select(0, keep_indices_tensor);
    /*
        index_select(dim, index) 语法解析
        dim = 0：表示我们要沿着第 0 维度（即“行”或“数据点个数”的维度）进行挑选。
        index：就是刚才上传到 GPU 的那份筛选索引
    */
    auto filtered_colors = colors.index_select(0, keep_indices_tensor);
    auto filtered_depths_in_rsp_frame = depths_in_rsp_frame.index_select(0, keep_indices_tensor);
    auto filtered_pixels = pixels.index_select(0, keep_indices_tensor);


    // 基于 GPU 的几何与遮挡剔除
    int H = viewpoint_cam->image_height_, W = viewpoint_cam->image_width_;
    auto filter = [H, W, &rendered_alpha](const torch::Tensor& points, 
                                        const torch::Tensor& colors, 
                                        const torch::Tensor& depths_in_rsp_frame, 
                                        const torch::Tensor& pixels) 
    {
        // 1. 视锥体范围检查 (Frustum Culling)
        // 确保投影后的点坐标落在范围【（0，W）*（0，H）】，落在外面的点是无效的，剔除
        auto in_image = (pixels.index({torch::indexing::Slice(), 0}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 0}) < W) &     // 所有x点
                        (pixels.index({torch::indexing::Slice(), 1}) >= 0) &    // 所有y点
                        (pixels.index({torch::indexing::Slice(), 1}) < H);  // (n) bool
        
        // 2. 深度有效性检查
        auto positive_depth = depths_in_rsp_frame > 0;
        // 3. 遮挡检查 (Occlusion Culling) —— 最关键的一步！
        auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);  // coords 坐标
        auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
        // 查阅当前的“渲染透明度图”
        auto opaque = rendered_alpha.index({y_coords, x_coords}) < 0.99;  // (n) bool

        // 综合判断，将上述三个条件取交集，再进行判断
        auto valid_flag = torch::logical_and(torch::logical_and(in_image, positive_depth), opaque);
        auto filtered_points = points.index({valid_flag, torch::indexing::Slice()});
        auto filtered_colors = colors.index({valid_flag, torch::indexing::Slice()});
        auto filtered_depths = depths_in_rsp_frame.index({valid_flag});
        // return std::make_tuple(filtered_points, filtered_colors, filtered_depths);
        // ================Probabilistic Gaussian-LIC===================== version 2
       // 同时返回过滤后的像素坐标，用于后续的不确定性采样
        auto filtered_pixel_coords = pixels.index({valid_flag, torch::indexing::Slice()});
        return std::make_tuple(filtered_points, filtered_colors, filtered_depths, filtered_pixel_coords);
        // ================Probabilistic Gaussian-LIC===================== version 2
    };

    // auto filtered_pkg = filter(points, colors, depths_in_rsp_frame, pixels);  // 调用上面的lambda表达式
    auto filtered_pkg = filter(filtered_points, filtered_colors, filtered_depths_in_rsp_frame, filtered_pixels);
    
    /// densification
    torch::Tensor fused_point_cloud = std::get<0>(filtered_pkg);  // (n, 3)
    torch::Tensor fused_color = RGB2SH(std::get<1>(filtered_pkg));
    torch::Tensor fused_depths = std::get<2>(filtered_pkg);       // (n,)
    torch::Tensor fused_pixel_coords = std::get<3>(filtered_pkg); // (n, 2) - 新增：像素坐标
    int num = fused_point_cloud.size(0);
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    torch::Tensor features_dc = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();  // (n, 1, 3)
    torch::Tensor features_rest = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();  // (n, 15, 3)
    // torch::Tensor scales = torch::log(pc->scaling_scale_ * std::get<2>(filtered_pkg) / focal).unsqueeze(1).repeat({1, 3});  // (n, 3)
    // torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    // rots.index({torch::indexing::Slice(), 0}) = 1;
    // torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    // ================Probabilistic Gaussian-LIC===================== version 2
    // UWGI: 不确定性加权高斯初始化
    torch::Tensor scales;
    torch::Tensor opacities;
    torch::Tensor point_uncertainties;  // 用于存储每个高斯的不确定性
    
    // 基础尺度计算：s_base = log(scaling_scale * depth / focal)
    torch::Tensor base_scales = torch::log(pc->scaling_scale_ * fused_depths / focal);  // (n,)
    
   // 检查是否启用概率初始化且有不确定性数据
    bool use_uwgi = pc->enable_probabilistic_ && 
                    geo_output != nullptr && 
                    geo_output->uncertainty_map.numel() > 0;
    
    if (use_uwgi) {
        // 从不确定性图中采样每个点的不确定性值
        // uncertainty_map: (H, W), fused_pixel_coords: (n, 2) where [x, y]
        auto uncertainty_map = geo_output->uncertainty_map;  // 已经在 CUDA 上
        
        // 获取像素坐标（注意：x是列，y是行）
        auto x_coords = fused_pixel_coords.index({torch::indexing::Slice(), 0}).to(torch::kLong);
        auto y_coords = fused_pixel_coords.index({torch::indexing::Slice(), 1}).to(torch::kLong);
        
        // 边界检查和裁剪
        int unc_H = uncertainty_map.size(0);
        int unc_W = uncertainty_map.size(1);
        x_coords = x_coords.clamp(0, unc_W - 1);
        y_coords = y_coords.clamp(0, unc_H - 1);
        
        // 从不确定性图中采样：uncertainty[y, x]
        point_uncertainties = uncertainty_map.index({y_coords, x_coords});  // (n,)
        
        // UWGI 公式实现:
        // 1. 深度方向拉长的各向异性缩放: s_z = s_base * (1 + β * σ)
        //    不确定性高的区域，高斯在深度方向拉长，增加覆盖范围
        float beta = pc->uncertainty_beta_;
        torch::Tensor scale_factor = 1.0f + beta * point_uncertainties;  // (n,)
        
        // 各向异性缩放：z方向（深度方向）根据不确定性拉长
        // scales[:, 0:2] = base_scales (xy方向)
        // scales[:, 2] = base_scales * scale_factor (z方向)
        torch::Tensor scale_xy = base_scales.unsqueeze(1).repeat({1, 2});  // (n, 2)
        torch::Tensor scale_z = (base_scales * scale_factor).unsqueeze(1);  // (n, 1)
        scales = torch::cat({scale_xy, scale_z}, 1);  // (n, 3)
        
        // 2. 不透明度调制: α_init = α_base * (1 - σ)
        //    不确定性高的点更透明，让网络有更多调整空间
        float alpha_base = 0.1f;
        torch::Tensor adjusted_opacity = alpha_base * (1.0f - point_uncertainties);  // (n,)
        // 裁剪到有效范围，避免数值问题
        adjusted_opacity = adjusted_opacity.clamp(0.01f, 0.99f);
        opacities = general_utils::inverse_sigmoid(adjusted_opacity.unsqueeze(1));  // (n, 1)
        
        std::cout << "\033[1;35m[UWGI] β=" << beta 
                  << ", mean_uncertainty=" << point_uncertainties.mean().item<float>() 
                  << "\033[0m ";
    } else {
        // 回退到原始的各向同性初始化
        scales = base_scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
        opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)
        point_uncertainties = torch::zeros({num}, torch::kFloat32).cuda();  // 默认不确定性为0
    }
    // ================Probabilistic Gaussian-LIC===================== version 2
    
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    

    // 把这些结果传入到稠密化重建之中，修改新的索引值为添加后的位置。函数在上面实现。
    pc->densificationPostfix(fused_point_cloud, features_dc, features_rest, opacities, scales, rots,point_uncertainties); 

    // ================Probabilistic Gaussian-LIC===================== version 2
    // // 更新高斯的不确定性属性
    // if (pc->enable_probabilistic_) {
    //     if (pc->uncertainty_.numel() == 0) {
    //         pc->uncertainty_ = point_uncertainties;
    //     } else {
    //         pc->uncertainty_ = torch::cat({pc->uncertainty_, point_uncertainties}, 0);
    //     }
    // }
    // ================Probabilistic Gaussian-LIC===================== version 2

    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000 
              << "k GS" << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc,const GeometryOutput* geo_output)
{
    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    std::vector<int> opt_list;
    int max_iters = 100;

    int train_camera_num = dataset->train_cameras_.size();
    std::vector<int> all_list(train_camera_num);
    std::iota(all_list.begin(), all_list.end(), 0);//从起始值0开始，为容器中的每个元素依次赋值为0, 1, 2, 3, ...

    std::random_device rd;
    std::mt19937 gen(rd());
    if (train_camera_num <= max_iters)  // 如果训练集的相机选择数量小于训练集，那么就直接使用全部的作为优化测试对象
    {
        opt_list = all_list;
    }
    else
    {
        // std::sample是 C++17 引入的算法，用于从序列中无放回随机抽样
        /*
            all_list.begin(), all_list.end()：输入序列的范围
            std::back_inserter(opt_list)：输出迭代器
            max_iters：要抽取的样本数量
            gen：随机数生成器
        */
        std::sample(all_list.begin(), all_list.end(), 
                    std::back_inserter(opt_list), max_iters, gen);
    } 
    std::shuffle(opt_list.begin(), opt_list.end(), gen);   // 随机打乱顺序
    // 🔧 修复：确保当前关键帧一定在采样列表中
    int last_idx = train_camera_num - 1;
    if (std::find(opt_list.begin(), opt_list.end(), last_idx) == opt_list.end()) {
        // 如果最后一帧不在列表中，替换第一个元素
        if (!opt_list.empty()) {
            opt_list[0] = last_idx;
        }
    }
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    // ================Mip_Splat_part===================== version 1
    // optimize函数内部，在渲染循环之前添加3D Filter更新
    // 每隔filter_3d_update_freq_次迭代更新一次最大采样率

    static int global_iteration_count = 0;
    global_iteration_count ++;

    if(pc->is3DFilterEnable() && global_iteration_count % pc->filter_3d_update_freq_ == 0)
    {
        pc->computeMaxSamplingRate(dataset->train_cameras_);
    }


    // ================Mip_Splat_part===================== version 1

    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    for (int idx : opt_list)
    {
        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = dataset->train_cameras_[idx];   // 找到随机采样的相机视角
        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        //这段代码构成了 3D Gaussian Splatting 优化循环中的 前向传播（Forward Pass） 与 损失计算（Loss Computation） 环节。
        pc->t_start_ = std::chrono::steady_clock::now();

        // 修改为
        auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_);
        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_final_T = std::get<1>(render_pkg);
        auto rendered_depth = std::get<2>(render_pkg);  // ← 新增：获取深度
        auto screenspace_points = std::get<3>(render_pkg);
        auto visible = std::get<4>(render_pkg);
        auto radii = std::get<5>(render_pkg);

        auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);  // 像素级光度损失（L1 Loss）
        float lambda_dssim = pc->lambda_dssim_;
        torch::Tensor ssim_value;
        //结构相似性损失（SSIM Loss）与维度适配
        torch::Tensor rendered_image_unsq = rendered_image.unsqueeze(0);  // 在第0维度增加一个维度
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        /* unsqueezed说明：
        rendered_image 的形状通常是 [C, H, W]（例如 [3, 720, 1280]）。
        然而，标准的 SSIM 算子（通常基于卷积实现）期望的输入形状是 Batch 模式，即 [B, C, H, W]。
        unsqueeze(0)：在第 0 维增加一个维度，将形状变为 [1, 3, 720, 1280]，从而满足 API 的输入要求。
        */
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);  // 作者重写了ssim函数。F12查看。
        auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);   // 3DGS标准的误差计算公式
        // ================Probabilistic Gaussian-LIC===================== ↓
        // 添加几何约束损失
        // if(pc->enable_probabilistic_ && dataset->current_dense_depth_.numel() > 0 && dataset->current_uncertainty_map_.numel()>0)
        // {
        //     //从渲染结果估计深度
        //     // 注意：这里简化处理，实际需要修改rasterizer输出深度图！！！！！！
        //     // 或者使用1-final_T作为深度的代理

        //     // 获取先验深度和不确定性
        //     auto prior_depth = dataset->current_dense_depth_;
        //     auto uncertainty_map = dataset->current_uncertainty_map_;

        //     // 这里需要rendered_depth,简化方案是使用额外的深度渲染pass
        //     // 或者修改rasterizer直接输出深度
        //     // 暂时使用placeholder

        //     // if(false){  // TODO:当实现深度渲染后启用
        //     if(pc->enable_probabilistic_ && 
        //             dataset->current_dense_depth_.numel() > 0 && 
        //             dataset->current_uncertainty_map_.numel() > 0 &&
        //             rendered_depth.numel() > 0){
        //     // ✓ 直接使用外部的 rendered_depth，不要重新声明！
            
        //     auto L_geo = loss_utils::geometric_constraint_loss(
        //         rendered_depth,  // ← 现在会使用第1248行获取的正确值
        //         prior_depth,
        //         uncertainty_map,
        //         static_cast<float>(dataset->fx_),
        //         static_cast<float>(dataset->fy_),
        //         pc->lambda_depth_,
        //         pc->lambda_normal_
        //     );
            
        //     loss = loss + L_geo;
        // }
        // }

        // ================Probabilistic Gaussian-LIC===================== 
    // 添加几何约束损失 - 只对最新关键帧应用
    // 注意: geo_output 只包含当前关键帧的深度/不确定性信息
    // std::cout << "[DEBUG] rendered_depth shape: " << rendered_depth.sizes() << std::endl;
    // std::cout << "[DEBUG] prior_depth shape: " << prior_depth.sizes() << std::endl;
    // ================Probabilistic Gaussian-LIC===================== 
                            // 原版修改1 // 添加几何约束损失 - 只对最新关键帧应用    原版修改1
                            // bool is_current_keyframe = (idx == static_cast<int>(dataset->train_cameras_.size()) - 1);

                            // if(pc->enable_probabilistic_ && 
                            //    is_current_keyframe &&
                            //    geo_output != nullptr && 
                            //    geo_output->dense_depth.numel() > 0 &&
                            //    geo_output->uncertainty_map.numel() > 0 &&
                            //    rendered_depth.numel() > 0) {

                            //     // 直接使用 geo_output 的数据
                            //     auto prior_depth = geo_output->dense_depth.to(torch::kCUDA);
                            //     auto uncertainty_map = geo_output->uncertainty_map.to(torch::kCUDA);
                                
                            //     // 调试输出
                            //     std::cout << "[DEBUG] rendered_depth shape: " << rendered_depth.sizes() 
                            //               << ", prior_depth shape: " << prior_depth.sizes() << std::endl;
                                
                            //     // 获取尺寸 - 使用正确的成员名
                            //     int64_t H = viewpoint_cam->image_height_;
                            //     int64_t W = viewpoint_cam->image_width_;
                                
                            //     // reshape rendered_depth - 使用正确的语法
                            //     auto rendered_depth_2d = rendered_depth.view(torch::IntArrayRef({H, W}));
                                
                            //     auto L_geo = loss_utils::geometric_constraint_loss(
                            //         rendered_depth_2d,
                            //         prior_depth,
                            //         uncertainty_map,
                            //         static_cast<float>(dataset->fx_),
                            //         static_cast<float>(dataset->fy_),
                            //         pc->lambda_depth_,
                            //         pc->lambda_normal_
                            //     );
                            
                            //     // 使用正确的模板语法
                            //     float geo_loss_val = L_geo.item<float>();
                            //     std::cout << "[Geo Loss] L_geo=" << geo_loss_val
                            //               << " lambda_d=" << pc->lambda_depth_
                            //               << " lambda_n=" << pc->lambda_normal_ << std::endl;
                                
                            //     loss = loss + L_geo;
                            // }

        // 修改后版本2 
        //  🔧 修改：对所有有深度先验的视图应用几何约束损失（而非仅当前关键帧）
        if(pc->enable_probabilistic_ && 
        dataset->hasPriorDepth(idx) &&  // ← 关键修改：检查该视图是否有深度先验
        dataset->hasSparseDepth(idx) &&  // ⬅️ 新增检查
        rendered_depth.numel() > 0) {

            // 从 Dataset 缓存中获取深度先验（而非 geo_output）
            auto prior_depth = dataset->prior_depths_.at(idx).to(torch::kCUDA);
            auto uncertainty_map = dataset->uncertainty_maps_.at(idx).to(torch::kCUDA);
            auto sparse_depth = dataset->sparse_depths_.at(idx).to(torch::kCUDA);   // ⬅️ 原始 LiDAR 深度
    
            
            // 获取尺寸
            int64_t H = viewpoint_cam->image_height_;
            int64_t W = viewpoint_cam->image_width_;
            
            // reshape rendered_depth
            auto rendered_depth_2d = rendered_depth.view(torch::IntArrayRef({H, W}));
            
            // auto L_geo = loss_utils::geometric_constraint_loss(
            //     rendered_depth_2d,
            //     prior_depth,
            //     uncertainty_map,
            //     static_cast<float>(dataset->fx_),
            //     static_cast<float>(dataset->fy_),
            //     pc->lambda_depth_,
            //     pc->lambda_normal_
            // );

            // ⬅️ 使用新的混合几何约束损失
            auto L_geo = loss_utils::hybrid_geometric_constraint_loss(
                rendered_depth_2d,
                prior_depth,           // SGSNet 补全深度
                uncertainty_map,
                sparse_depth,          // ⬅️ 原始 LiDAR 稀疏深度
                static_cast<float>(dataset->fx_),
                static_cast<float>(dataset->fy_),
                pc->lambda_lidar_,     // ⬅️ 新参数：LiDAR 权重 (建议 1.0)
                pc->lambda_sgs_,       // ⬅️ 新参数：SGSNet 权重 (建议 0.1)
                pc->lambda_normal_,    // 法向权重
                true                   // 启用尺度对齐
            );

            loss = loss + L_geo;
            
            // 调试输出（可选，建议只在当前关键帧输出以避免刷屏）
            if (idx == static_cast<int>(dataset->train_cameras_.size()) - 1) {
                float geo_loss_val = L_geo.item<float>();
                std::cout << "[DEBUG] rendered_depth shape: " << rendered_depth_2d.sizes() 
                        << ", prior_depth shape: " << prior_depth.sizes() << std::endl;
                // std::cout << "[Geo Loss] L_geo=" << geo_loss_val
                //         << " lambda_d=" << pc->lambda_depth_
                //         << " lambda_n=" << pc->lambda_normal_ << std::endl;
                std::cout << "[Hybrid Geo Loss] L_geo=" << geo_loss_val
                  << " lambda_lidar=" << pc->lambda_lidar_
                  << " lambda_sgs=" << pc->lambda_sgs_
                  << " lambda_n=" << pc->lambda_normal_ << std::endl;
            }
        // 测试代码
        // std::cout << "[DEBUG] rendered_depth requires_grad: " << rendered_depth.requires_grad() << std::endl;
        // std::cout << "[DEBUG] prior_depth requires_grad: " << prior_depth.requires_grad() << std::endl;
        // std::cout << "[DEBUG] L_geo requires_grad: " << L_geo.requires_grad() << std::endl;
        if (idx == static_cast<int>(dataset->train_cameras_.size()) - 1) {
            float total_loss = loss.item<float>();
            float l1_val = Ll1.item<float>();
            float ssim_val = ssim_value.item<float>();
            float geo_val = L_geo.item<float>();
            
            float photometric = (1.0 - lambda_dssim) * l1_val + lambda_dssim * (1.0 - ssim_val);
            
            std::cout << "[LOSS BREAKDOWN] "
                    << "Total=" << total_loss 
                    << " | Photometric=" << photometric
                    << " (L1=" << l1_val << ", SSIM=" << ssim_val << ")"
                    << " | Geo=" << geo_val 
                    << " | Geo%=" << (geo_val / total_loss * 100) << "%" << std::endl;
        }
        }

        // ================Probabilistic Gaussian-LIC=====================

        // 渲染深度调试输出
        std::cout << "Rendered depth: min=" << rendered_depth.min().item<float>() 
                  << " max=" << rendered_depth.max().item<float>() << std::endl;

    

        // ================Probabilistic Gaussian-LIC=====================

        // // 在 optimize() 中添加验证代码
        // // auto rendered_depth = std::get<2>(render_pkg);
        // std::cout << "Rendered depth: min=" << rendered_depth.min().item<float>() 
        //         << " max=" << rendered_depth.max().item<float>() << std::endl;
                
        // ================Probabilistic Gaussian-LIC===================== ↑
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        

        // 反向传播
        pc->t_start_ = std::chrono::steady_clock::now();
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();


        //稀疏优化器步进 (Sparse Optimizer Step)
        pc->t_start_ = std::chrono::steady_clock::now();
        // auto visible = std::get<3>(render_pkg);  // 只有在视锥范围之内且对最终图像有贡献的高斯点其对应位才为True。否则会遍历所有的点！太浪费了！
        updated_num += visible.sum().item<int>();
        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0)); // 将visible的点设置为true
        pc->sparse_optimizer_->step();               // 执行稀疏更新
        pc->sparse_optimizer_->zero_grad(true);  //高效清空梯度
        if (pc->apply_exposure_)  // 曝光参数优化
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }

    return updated_num / opt_list.size();
}

//  ================Mip_Splat_part===================== version 1
/**
 * @brief 计算每个高斯的最大采样率 (用于3D Smoothing Filter)
 * 
 * 基于奈奎斯特采样定理:
 * ν_k = max_{n=1}^{N} (I_n(p_k) * f_n / d_n)
 * 
 * 其中:
 * - I_n(p_k): 指示函数，高斯k是否在相机n的视锥内
 * - f_n: 相机n的焦距
 * - d_n: 高斯k到相机n的深度
 */

void GaussianModel::computeMaxSamplingRate(const std::vector<std::shared_ptr<Camera>>& cameras)
{
    torch::NoGradGuard no_grad;

    int num_gaussians = xyz_.size(0);
    if(num_gaussians == 0) return ;

    // 获取高斯位置
    auto xyz_cpu = xyz_.detach().cpu();
    auto xyz_accessor = xyz_cpu.accessor<float,2>();

    // 存储每一个高斯的最大采样率
    std::vector<float>max_rates(num_gaussians,0.0f);

    // 遍历所有相机
    for(const auto &camera:cameras){
        // 获取相机参数
        Eigen::Matrix3d R_cw = camera->R_cw_;
        Eigen::Vector3d t_cw = camera->t_cw_;
        double fx = camera->fx_;
        double fy = camera->fy_;
        double f = (fx+fy)/2.0;

        int W = camera->image_width_;
        int H = camera->image_height_;

        // 对每一个高斯计算采样率
        # pragma omp parallel for
        for(int i = 0;i<num_gaussians;++i)
        {
            // 获取高斯世界坐标
            Eigen::Vector3d p_w(xyz_accessor[i][0],xyz_accessor[i][1],xyz_accessor[i][2]);

            // 转换到相机坐标系
            Eigen::Vector3d p_c = R_cw*p_w+t_cw;

            // 检查是否在相机前方
            double depth = p_c.z();
            if(depth<=0.1) continue;

            // 检查是否在视锥内
            double x_norm = p_c.x()/depth;
            double y_norm = p_c.y()/depth;
            double lim_x = 1.3*W/(2.0*fx);
            double lim_y = 1.3*H/(2.0*fy);

            if(std::abs(x_norm)>lim_x || std::abs(y_norm)>lim_y) continue;

            // 计算采样率: ν = f / d
            // 物理意义: 图像上1像素对应3D空间的采样间隔为 d/f
            //          采样频率为其倒数 f/d

            double sample_rate = f/depth;
            
            # pragma omp critical
            {
                max_rates[i] = std::max(max_rates[i],static_cast<float>(sample_rate));
            }
            
        }
    }

    // 转化为CUDA tensor
    max_sampling_rate_ = torch::from_blob(max_rates.data(),{num_gaussians},
                                            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(torch::kCUDA);

    
    // 设置最小值，防止除零
    max_sampling_rate_ = torch::clamp_min(max_sampling_rate_,1.0f);

    // 统计信息
    float min_rate = max_sampling_rate_.min().template item<float>();
    float max_rate = max_sampling_rate_.max().template item<float>();
    float mean_rate = max_sampling_rate_.mean().template item<float>();

    std::cout<<std::fixed<<std::setprecision(2)
            << "\033[1;35m [3D Filter] Sampling rate: min=" << min_rate 
              << ", max=" << max_rate << ", mean=" << mean_rate << "\033[0m" << std::endl;
}
/**
 * @brief 获取应用3D滤波器后的正则化缩放参数
 * 
 * 公式: Σ_reg = Σ + s/ν² * I
 * 
 * 这给3D高斯的尺寸加了一个"硬底":
 * - 深度d越小（离得越近），ν越大，分母越大，加上的底越小（允许高斯更小）
 * - 深度d越大（离得越远），必须加上一个更大的底，强迫高斯变"胖"
 */

torch::Tensor GaussianModel::getRegularizedScaling(){

    if(!apply_3d_filter_ || max_sampling_rate_.numel() == 0)
    {
        // 如果没有启动3d滤波，返回原始参数
        return getScaling();
    }

    // 获取原始缩放（N，3）
    torch::Tensor scales = getScaling();
    if(!apply_3d_filter_ || max_sampling_rate_.numel() == 0)
    {
        return scales;
    }
    // ================Mip_Splat_part===================== version 1
    // 关键修复，检查大小是否匹配
    int num_gaussians = scales.size(0);
    if(max_sampling_rate_.size(0) != num_gaussians)
    {
        // 大小不匹配时，返回原始缩放参数，等待下次computeMaxSamplingRate 更新）
        // 可选：打印警告
        std::cerr << "[3D Filter Warning] Size mismatch: scales=" << num_gaussians 
                  << ", sampling_rate=" << max_sampling_rate_.size(0) << std::endl;
        
        return scales;
    }

    // ================Mip_Splat_part===================== version 1
    

    // 计算滤波器方差: filter_var = s / ν²
    // 形状: (N,) -> (N, 1) 用于广播
    torch::Tensor nu_sq = max_sampling_rate_*max_sampling_rate_;
    torch::Tensor filter_var = filter_3d_scale_/nu_sq;
    filter_var = filter_var.unsqueeze(1); // (N,1)

    // 原始方差 (scales存储的是标准差σ)
    torch::Tensor scales_sq = scales*scales; // σ² (N, 3)

    // 正则化方差 σ_reg² = σ² + filter_var
    torch::Tensor scales_reg_sq = scales_sq+filter_var;

    // 正则化标准差
    torch::Tensor scales_reg = torch::sqrt(scales_reg_sq); // (N,3)

    return scales_reg;
}

//  ================Mip_Splat_part===================== version 1
void evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset, // 数据集，包含训练/测试相机视角及其对应的真值图像
                           std::shared_ptr<GaussianModel>& pc,      // 高斯模型（Point Cloud），存储所有高斯球的参数
                           const std::string& result_path,          // 结果保存路径
                           const std::string& lpips_path)           // LPIPS预训练模型（TorchScript格式）的路径
{
    std::cout << "\n     🎉 Evaluate Visual Quality 🎉\n";
    std::cout << "\n        [Number of Final Gaussians] " << pc->getXYZ().size(0) << std::endl;  // 输出点的个数，size（0）是第0维的长度，即表示点的数量

    // 如果结果目录已存在，则递归删除整个目录，然后重新创价目录
    // if (fs::exists(result_path)) fs::remove_all(result_path);
    // fs::create_directories(result_path);

    // 创建两个子目录保存
    // std::string render_dir_path = result_path + "/render";
    // fs::create_directories(render_dir_path);
    // std::string gt_dir_path = result_path + "/gt";
    // fs::create_directories(gt_dir_path);

    //=================根据时间的保存路径========================↓
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* now_tm = std::localtime(&now_time);

    std::ostringstream time_ss;

    time_ss<<std::setfill('0')
            <<std::setw(2)<<(now_tm->tm_mon+1)<<"-"
            <<std::setw(2)<<now_tm->tm_mday<<"-"
            <<std::setw(2)<<now_tm->tm_hour<<"-"
            <<std::setw(2)<<now_tm->tm_min;

    std::string timestamp_dir = time_ss.str();

    // 创建带时间戳的路径
    std::string timestamp_path = result_path+"/"+timestamp_dir;
    fs::create_directories(timestamp_path);

    std::string render_dir_path = timestamp_path + "/render";
    fs::create_directories(render_dir_path);
    std::string gt_dir_path = timestamp_path + "/gt";
    fs::create_directories(gt_dir_path);

    //=================根据时间的保存路径========================↑

    // 结果＋时间

    // 背景设置与LPIPS模型加载。 这个背景会在渲染时用于填充透明区域。
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::jit::script::Module m_lpips;
    try 
    {
        m_lpips = torch::jit::load(lpips_path + "/lpips_alex.pt");
        m_lpips.to(torch::kCUDA);
        /*
            torch::jit::load：加载预先用Python导出的TorchScript模型（.pt文件）
            lpips_alex.pt：使用AlexNet作为backbone的LPIPS模型
            .to(torch::kCUDA)：将模型移到GPU上
            LPIPS（Learned Perceptual Image Patch Similarity）是基于深度学习的感知相似度指标，比PSNR/SSIM更符合人眼主观感受。
        */
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
    }

    {   // 训练视角评估 。用花括号创建一个独立的作用域，避免变量名冲突，初始化三个累加器。
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& train_camera : dataset->train_cameras_)  // 遍历所有训练视角的相机
        {
            auto render_pkg = render(train_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            /*
            调用 render() 函数，从当前相机视角渲染高斯模型
            render_pkg 是一个 tuple，std::get<0> 取出渲染图像
            .clamp(0, 1)：将像素值裁剪到 [0, 1] 范围（防止数值溢出）
            */
           // 获取该视角真是图像，移到GPU并裁剪。
            auto gt_image = train_camera->original_image_.cuda().clamp(0, 1); 
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            /*
            *PSNR（峰值信噪比）**计算：
                loss_utils::psnr()：计算PSNR值
                .mean()：如果返回多通道结果，取平均
                .item<double>()：将单元素Tensor转换为C++ double标量
                PSNR越高越好，典型值：20-40 dB
            */
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            /*
            **SSIM（结构相似性）**计算：
                衡量图像的结构、亮度、对比度相似程度
                范围 [0, 1]，越接近1越好
            */
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            /*
            LPIPS计算：
                IValue：TorchScript的通用值类型，用于跨语言传参
                unsqueeze(0)：添加batch维度，从 [C,H,W] 变为 [1,C,H,W]
                m_lpips.forward(inputs)：执行前向推理
                LPIPS越低越好（表示感知差异越小）
            */
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            // 获取图像尺寸主备保存图像。注意Tensor的形状是【C，H，W】所以 size（1）= H size（2） = W
            int H = rendered_image.size(1), W = rendered_image.size(2);
            //Tensor → OpenCV Mat 的转换流程：
            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + train_camera->image_name_, a_img);
            /*
            1.  to(torch::kCPU)GPU → CPU
            2.  permute({1, 2, 0})[C,H,W] → [H,W,C]（OpenCV格式）
            3.  contiguous()确保内存连续（permute后可能不连续）
            4.  mul(255)[0,1] → [0,255]
            5.  clamp(0, 255)裁剪到有效范围
            6.  to(torch::kU8)float32 → uint8
                    COLOR_RGB2BGR：PyTorch/3DGS用RGB，OpenCV用BGR，需要转换
                    保存渲染图像到 render/ 目录
            */

            // 同样过程，真值保存在gt目录之下
            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + train_camera->image_name_, b_img);
        }
        // 打印输出训练视角的平均指标
        psnrs /= dataset->train_cameras_.size();
        ssims /= dataset->train_cameras_.size();
        lpipss /= dataset->train_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [Training View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [Training View LPIPS] " << lpipss << std::endl;
    }
    {   // 这一段是测试视角的评估，其实和训练视角是一样的，只是便利的数据集该文了test_cameras_
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        for (const auto& test_camera : dataset->test_cameras_)
        {
            auto render_pkg = render(test_camera, pc, bg, pc->apply_exposure_);
            auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
            auto gt_image = test_camera->original_image_.cuda().clamp(0, 1);
            double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
            double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(rendered_image.unsqueeze(0));
            inputs.push_back(gt_image.unsqueeze(0));
            double lpips = m_lpips.forward(inputs).toTensor().item<double>();
            psnrs += psnr;
            ssims += ssim;
            lpipss += lpips;

            int H = rendered_image.size(1), W = rendered_image.size(2);

            torch::Tensor a_cpu = rendered_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            a_cpu = a_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat a_img(H, W, CV_8UC3, a_cpu.data_ptr<uint8_t>());
            cv::cvtColor(a_img, a_img, cv::COLOR_RGB2BGR);
            cv::imwrite(render_dir_path + "/" + test_camera->image_name_, a_img);

            torch::Tensor b_cpu = gt_image.to(torch::kCPU).permute({1, 2, 0}).contiguous();
            b_cpu = b_cpu.mul(255).clamp(0, 255).to(torch::kU8);
            cv::Mat b_img(H, W, CV_8UC3, b_cpu.data_ptr<uint8_t>());
            cv::cvtColor(b_img, b_img, cv::COLOR_RGB2BGR);
            cv::imwrite(gt_dir_path + "/" + test_camera->image_name_, b_img);
        }
        psnrs /= dataset->test_cameras_.size();
        ssims /= dataset->test_cameras_.size();
        lpipss /= dataset->test_cameras_.size();
        std::cout << std::fixed << std::setprecision(2) << "        [In-Sequence Novel View PSNR] " << psnrs << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View SSIM] " << ssims << std::endl;
        std::cout << std::fixed << std::setprecision(3) << "        [In-Sequence Novel View LPIPS] " << lpipss << std::endl;
    }
}

// ================Probabilistic Gaussian-LIC===================== version 2
torch::Tensor GaussianModel::getUncertainty()  // ✅ 改为 getUncertainty，返回类型改为 torch::Tensor
{
    if(uncertainty_.numel() == 0)
    {
        // 如果未初始化，返回全0（完全确信）
        return torch::zeros({xyz_.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }
    return uncertainty_;  // ✅ 取消注释
}

void GaussianModel::setUncertainty(const torch::Tensor& uncertainty)
{
    uncertainty_ = uncertainty.to(torch::kCUDA);
}

torch::Tensor GaussianModel::getUncertaintyModulatedScaling()
{
    torch::Tensor scales = getScaling(); // 获取原始缩放（N，3）
    if(!enable_probabilistic_ || uncertainty_.numel() == 0){
        return scales;
    }

    // 检查匹配大小
    if(uncertainty_.size(0) != scales.size(0))
    {
        return scales;
    }

    // 不确定性越高，scale越大（高斯更模糊）
    // scale_modulated = scale * (1 + uncertainty * factor)
    torch::Tensor uncertainty_expanded = uncertainty_.unsqueeze(1);// (N, 1)
    torch::Tensor scale_factor = 1.0f + uncertainty_expanded * uncertainty_beta_;


    return scales * scale_factor; // (N,3)
     
}

void GaussianModel::probabilisticInitialize(
    const std::shared_ptr<Dataset>& dataset,
    const torch::Tensor& dense_depth,
    const torch::Tensor& uncertainty_map
)
{
    torch::NoGradGuard no_grad;

    // 获取相机内参
    float fx = static_cast<float>(dataset->fx_);
    float fy = static_cast<float>(dataset->fy_);
    float cx = static_cast<float>(dataset->cx_);
    float cy = static_cast<float>(dataset->cy_);

    float focal = (fx + fy)/2.0f;

    int H = dense_depth.size(0);
    int W = dense_depth.size(1);

    // 将深度和不确定性转到cpu进行处理
    auto depth_cpu = dense_depth.to(torch::kCPU);
    auto uncertainty_cpu = uncertainty_map.to(torch::kCPU);
    auto depth_accessor = depth_cpu.accessor<float,2>();
    auto uncertainty_accessor = uncertainty_cpu.accessor<float,2>();

    // 获取最新相机位姿
    Eigen::Matrix3d R_wc = dataset->R_wc_.back();
    Eigen::Vector3d t_wc = dataset->t_wc_.back();

    //收集满足置信度阈值的点
    std::vector<Eigen::Vector3d> valid_points;
    std::vector<Eigen::Vector3d> valid_colors;
    std::vector<float> valid_depths;
    std::vector<float> valid_uncertainties;

    // 降采样因子（避免点云过密）
    int sample_step = 4;  // 降采样因子    原本是4，为了增加高斯点数我改成了2，增加了4倍
    for(int v = 0;v<H;v+=sample_step){
        for(int u = 0; u < W;u += sample_step)
        {
            float depth = depth_accessor[v][u];
            float uncertainty = uncertainty_accessor[v][u];

            // 置信度门控：值初始化不确定性低于阈值的点
            if(depth <= 0.1f || uncertainty > uncertainty_threshold_)
            { continue; } 

            // 反投影到相机坐标系
            float x_c = (u - cx)*depth/fx;
            float y_c = (v - cy)*depth/fy;
            float z_c = depth;

            // 转换到世界坐标系
            Eigen::Vector3d p_c(x_c,y_c,z_c);
            Eigen::Vector3d p_w = R_wc*p_c + t_wc;

            valid_points.push_back(p_w);
            valid_depths.push_back(depth);
            valid_uncertainties.push_back(uncertainty);

            // 颜色需要从图像获取（这里用灰色占位—）
            valid_colors.push_back(Eigen::Vector3d(0.5,0.5,0.5));
        }
    }

    if(valid_points.empty()){
        std::cerr << "[Probabilistic Init] No valid points found!" << std::endl;
        return;
    }

    int num = static_cast<int>(valid_points.size());
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);

    // 创建Tensor
    torch::Tensor fused_point_cloud = torch::zeros({num,3},torch::kFloat32).cuda();
    torch::Tensor features = torch::zeros({num,3,deg_2},torch::kFloat32).cuda();
    torch::Tensor scales = torch::zeros({num,3},torch::kFloat32).cuda();
    torch::Tensor rots = torch::zeros({num,4},torch::kFloat32).cuda();
    torch::Tensor opacities = torch::zeros({num,1},torch::kFloat32).cuda();
    torch::Tensor uncertainties = torch::zeros({num},torch::kFloat32).cuda();


    // 填充数据
    for (int i = 0; i < num; i++)
    {
        // 位置
        fused_point_cloud.index({i,0}) = valid_points[i].x();
        fused_point_cloud.index({i,1}) = valid_points[i].y();
        fused_point_cloud.index({i,2}) = valid_points[i].z();

        // 颜色（球谐系数）
        features.index({i,0,0}) = RGB2SH(valid_colors[i].x());
        features.index({i,1,0}) = RGB2SH(valid_colors[i].y());
        features.index({i,2,0}) = RGB2SH(valid_colors[i].z());

        // 尺度：基于深度和不确定性
        // s = log(scaling_scale * depth / focal) * (1 + uncertainty * factor)
        float base_scale = std::log(scaling_scale_ * valid_depths[i]/focal);
        float scale_factor = 1.0f + valid_uncertainties[i] * init_scale_uncertainty_factor_;
        float final_scale = base_scale * scale_factor;

        scales.index({i,0}) = final_scale;
        scales.index({i,1}) = final_scale;
        scales.index({i,2}) = final_scale;


        // 不确定性
        uncertainties.index({i}) = valid_uncertainties[i];

        // 不透明度：不确定的点更透明
        // opacity = sigmoid_inv(0.1 * (1 - uncertainty * factor))
        float base_opacity = 0.1f*(1.0f - valid_uncertainties[i] * init_opacity_uncertainty_factor_);
        base_opacity = std::max(0.01f,std::min(0.99f,base_opacity));
        opacities.index({i,0}) = std::log(base_opacity/(1.0f - base_opacity));

    }

    // 旋转初始化为单位四元数
    rots.index({torch::indexing::Slice(),0}) = 1;

    // 如果是第一次初始化
    if(!is_init_){
        this->xyz_ = fused_point_cloud.requires_grad_();
        this->features_dc_ = features.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(0,1)}).transpose(1,2).contiguous().requires_grad_();
        this->features_rest_ = features.index({torch::indexing::Slice(),torch::indexing::Slice(),torch::indexing::Slice(1,features.size(2))}).transpose(1,2).contiguous().requires_grad_();
        this->scaling_ = scales.requires_grad_();
        this->rotation_ = rots.requires_grad_();
        this->opacity_ = opacities.requires_grad_();
        this->uncertainty_ = uncertainties;

        GAUSSIAN_MODEL_TENSORS_TO_VEC

        is_init_ = true;
    }
    else{
        // 增量添加
        torch::Tensor new_features_dc = features.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
        torch::Tensor new_features_rest = features.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();
        
        densificationPostfix(fused_point_cloud, new_features_dc, new_features_rest, 
                            opacities, scales, rots,uncertainties);
        
        // 更新不确定性
        uncertainty_ = torch::cat({uncertainty_, uncertainties}, 0);

    }

    std::cout<<std::fixed<<std::setprecision(2)
             <<"\033[1;35m [Probabilitic Init] Added]"
             <<num<<"Gaussians threshold = "<<uncertainty_threshold_<<")\033[0m "<<std::endl;
    

}
// ================Probabilistic Gaussian-LIC===================== version 2
