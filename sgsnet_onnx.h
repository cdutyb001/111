#pragma once

// 包含这个 C API 头文件有时能解决依赖问题，虽然 CXX API 通常会自动包含它
#include <onnxruntime_c_api.h> 
#include <onnxruntime_cxx_api.h>


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class SGSNetONNX {
public:
    /**
     * @brief 初始化模型
     * @param model_path ONNX模型路径
     * @param use_cuda 是否尝试使用CUDA加速
     * @param height 初始图像高度
     * @param width 初始图象深度
     */
    SGSNetONNX(const std::string& model_path,int height,int width, bool use_cuda = true);
    SGSNetONNX(const std::string& model_path, bool use_cuda) ;
    [[deprecated("Use constructor with height/width parameters")]]
    ~SGSNetONNX() = default;

    /**
     * @brief 推理函数--version2 支持任意分辨率
     * @param rgb RGB图像 (H, W, 3), CV_8UC3
     * @param sparse_depth 稀疏深度图 (H, W, 1), CV_32FC1
     * @return pair<DenseDepth, Uncertainty> 都是 CV_32FC1
     */
    std::pair<cv::Mat, cv::Mat> infer(const cv::Mat& rgb, const cv::Mat& sparse_depth);


    void updateResolution(int height,int width);

    // Accessors
    int getHeight() const {return static_cast<int> (height_);}
    int getWidth() const {return static_cast<int>(width_);}
    bool isInitialized() const{return session_ !=nullptr;}

private:

    // 初始化相关
    void initializeSession(const std::string& model_path,bool use_cuda);
    void allocateBuffers();

    // ONNX Runtime 核心组件
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // 输入/输出节点名称
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    // 内存管理 (复用Buffer以提升性能)
    std::vector<float> input_rgb_values_;
    std::vector<float> input_depth_values_;
    
    // // 固定维度 (根据您的代码设定)
    // const int64_t height_ = 480;
    // const int64_t width_ = 640;

    // 动态维度
    int64_t height_;
    int64_t width_;
};