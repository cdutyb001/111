/*
 * Probabilistic Gaussian-LIC: Enhanced Geometry Head
 * Simplified version using bilateral filtering
 */


#pragma once

#include<opencv2/opencv.hpp>
#include<torch/torch.h>
#include<Eigen/Dense>
#include<memory>
#include<string>
#include<SGSNet/sgsnet_onnx.h>

struct GeometryOutput
{
   torch::Tensor dense_depth;       // SGSNet 补全后的稠密深度
   torch::Tensor uncertainty_map;   // 不确定性图
   torch::Tensor sparse_depth;      // ⬅️ 新增：原始 LiDAR 稀疏深度
   cv::Mat dense_depth_cv;
   cv::Mat uncertainty_cv;
   cv::Mat sparse_depth_cv;         // ⬅️ 新增
};


class GeometryHead{
    public:
    // ================Probabilistic Gaussian-LIC===================== version 2
    // 新增version2: 带SGSNet模型路径的构造函数
    GeometryHead(int height, int width,
                float fx, float fy, float cx, float cy,
                const std::string& sgsnet_model_path,
                bool use_cuda = true);
    // ================Probabilistic Gaussian-LIC===================== version 2
    
    // 旧接口兼容性保存
    GeometryHead(int height,int width,float fx,float fy,float cx,float cy);
      /**
     * @brief 处理一帧数据，生成稠密深度和不确定性
     * @param image RGB图像 (H, W, 3), float32, [0,1]
     * @param lidar_points LiDAR点云 (N, 3), 世界坐标系
     * @param R_cw 相机到世界的旋转矩阵的逆 (世界到相机)
     * @param t_cw 平移向量
     * @return GeometryOutput 包含稠密深度和不确定性图
     */

   ~GeometryHead() = default;  // 使用默认析构函数

    // 新增version 2：运行时切换方法
    void setUseSGSNet(bool use);
    bool isUsingSGSNet() const { return use_sgsnet_; }  // 内联实现
    

    GeometryOutput process(
        const cv::Mat &image,
        const std::vector<Eigen::Vector3d>& lidar_points,
        const Eigen::Matrix3d& R_cw,
        const Eigen::Vector3d& t_cw
    );

    private:
    // ================Probabilistic Gaussian-LIC===================== version 2
    //新增 version2 ：SGSNet 深度补全方法
    std::pair<cv::Mat,cv::Mat>sgsnetDensification(
        const cv::Mat& sparse_depth,
        const cv::Mat& rgb_image
    );
    std::unique_ptr<SGSNetONNX> sgsnet_engine_;
    bool use_sgsnet_ = true;
    // ================Probabilistic Gaussian-LIC===================== version 2

    // 投影lidar点到像素平面
    cv::Mat projectLidarToImage(
        const std::vector<Eigen::Vector3d>& lidar_points,
        const Eigen::Matrix3d& R_cw,
        const Eigen::Vector3d& t_cw
    );

    // 保留旧代码作为备用
    // 双边滤波稠密化
    cv::Mat bilateraDensification(
        const cv::Mat& sparsee_depth,
        const cv::Mat& rgb_image
    );
    // 计算不确定性图
    cv::Mat computeUncertainty(
        const cv::Mat& sparse_mask,
        const cv::Mat& rgb_image
    );

    int height_,width_;
    float fx_,fy_,cx_,cy_;

    // 滤波参数
    float sigma_color_ = 30.0f;             // 颜色空间标准差
    float sigma_space_ = 15.0f;             // 空间标准差
    int filter_interations_ = 3;            // 迭代次数
    float uncertainty_gamma_ = 0.1f;        // 不确定性衰减参数
};

