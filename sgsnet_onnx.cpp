#include "sgsnet_onnx.h"
#include <iostream>
#include <exception>

// ============================================================================
// 构造函数：推荐使用（带分辨率参数）
// ============================================================================
SGSNetONNX::SGSNetONNX(const std::string& model_path, int height,int width,bool use_cuda) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "SGSNet_Inference") ,height_(height),width_(width)
    {
        std::cout<<"[SGSNet] Initializing with resolution: "<<width<<"x"<<height<<std::endl;
        initializeSession(model_path,use_cuda);
        allocateBuffers();
    }
// ============================================================================
// 构造函数：兼容旧接口（已废弃）
// ============================================================================
SGSNetONNX::SGSNetONNX(const std::string& model_path, bool use_cuda) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "SGSNet_Inference"),
      height_(480),  // 默认值，为了兼容性保留
      width_(640)
{
    std::cerr << "[SGSNet] WARNING: Using deprecated constructor with default resolution 640x480." << std::endl;
    std::cerr << "[SGSNet] Please use SGSNetONNX(model_path, height, width, use_cuda) instead." << std::endl;
    initializeSession(model_path, use_cuda);
    allocateBuffers();
}  

// ============================================================================
// 初始化 ONNX Session
// ============================================================================    
void SGSNetONNX::initializeSession(const std::string& model_path,bool use_cuda)
    {

    // 1. 配置 Session
    session_options_.SetIntraOpNumThreads(1); // SLAM系统中通常限制单线程以免抢占资源
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. 配置 CUDA
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;  // kNextPowerOfTwo
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;


            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[SGSNet] CUDA execution provider enabled." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[SGSNet] Warning: Failed to enable CUDA, falling back to CPU. Error: " << e.what() << std::endl;
        }
    }

    // 3. 加载模型
    try {
        //由于Windows和Linux下wchar_t处理不同，通常建议在Linux下直接传char*
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        std::cout << "[SGSNet] Model loaded successfully: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "[SGSNet] Error loading model: " << e.what() << std::endl;
        throw;
    }

    // // 4. 预分配内存 (避免每帧 new/delete)
    // // RGB: 1 * 3 * H * W
    // input_rgb_values_.resize(1 * 3 * height_ * width_);
    // // Depth: 1 * 1 * H * W
    // input_depth_values_.resize(1 * 1 * height_ * width_);

    // 5. 定义节点名称 (必须与导出时的 input_names/output_names 一致)
    // 注意：这里的字符串必须在 Session 生命周期内有效
    static const char* in_names[] = {"rgb", "sparse_depth"};
    static const char* out_names[] = {"dense_depth", "uncertainty"}; // 对应您的代码要求
    
    input_node_names_ = std::vector<const char*>(std::begin(in_names), std::end(in_names));
    output_node_names_ = std::vector<const char*>(std::begin(out_names), std::end(out_names));
}

// ============================================================================
// 分配输入缓冲区
// ============================================================================
void SGSNetONNX::allocateBuffers(){
    const size_t image_size = static_cast<size_t>(height_*width_);

    // RGB: 1 * 3 * H * W
    input_rgb_values_.resize(3 * image_size);
    
    // Depth: 1 * 1 * H * W
    input_depth_values_.resize(image_size);
    
    std::cout << "[SGSNet] Buffers allocated for " << width_ << "x" << height_ 
              << " (RGB: " << input_rgb_values_.size() * sizeof(float) / 1024 << " KB, "
              << "Depth: " << input_depth_values_.size() * sizeof(float) / 1024 << " KB)" << std::endl;
}

// ============================================================================
// 更新分辨率（运行时重新分配缓冲区）
// ============================================================================
void SGSNetONNX::updateResolution(int height, int width) {
    if (height_ == height && width_ == width) {
        return;  // 分辨率未变化，无需重新分配
    }
    
    std::cout << "[SGSNet] Resolution changed: " << width_ << "x" << height_ 
              << " -> " << width << "x" << height << std::endl;
    
    height_ = height;
    width_ = width;
    allocateBuffers();
}

// ============================================================================
// 推理函数
// ============================================================================
std::pair<cv::Mat, cv::Mat> SGSNetONNX::infer(const cv::Mat& rgb, const cv::Mat& sparse_depth) {
    // ========== 输入验证 ==========
    if (rgb.empty() || sparse_depth.empty()) {
        std::cerr << "[SGSNet] Error: Empty input image!" << std::endl;
        return {};
    }
    
    if (rgb.type() != CV_8UC3) {
        std::cerr << "[SGSNet] Error: RGB image must be CV_8UC3, got type " << rgb.type() << std::endl;
        return {};
    }
    
    if (sparse_depth.type() != CV_32FC1) {
        std::cerr << "[SGSNet] Error: Sparse depth must be CV_32FC1, got type " << sparse_depth.type() << std::endl;
        return {};
    }
    
    if (rgb.rows != sparse_depth.rows || rgb.cols != sparse_depth.cols) {
        std::cerr << "[SGSNet] Error: RGB and depth size mismatch!" << std::endl;
        return {};
    }

    // ========== 动态分辨率处理 ==========
    const int input_height = rgb.rows;
    const int input_width = rgb.cols;
    
    if (input_height != height_ || input_width != width_) {
        updateResolution(input_height, input_width);
    }

    // ========== 步骤 1: 数据排布转换 (HWC -> CHW) ==========
    const int64_t image_size = height_ * width_;
    
    float* rgb_ptr = input_rgb_values_.data();
    float* depth_ptr = input_depth_values_.data();

    // 遍历像素，进行格式转换
    for (int64_t i = 0; i < image_size; ++i) {
        const int row = static_cast<int>(i / width_);
        const int col = static_cast<int>(i % width_);

        // 获取 RGB 像素 (OpenCV 默认 BGR)
        const cv::Vec3b& pixel = rgb.at<cv::Vec3b>(row, col);

        // 填充到 planar buffer (RRR...GGG...BBB...)
        // 并进行归一化 (0-255 -> 0.0-1.0)
        rgb_ptr[i]                  = pixel[2] / 255.0f; // R
        rgb_ptr[i + image_size]     = pixel[1] / 255.0f; // G
        rgb_ptr[i + image_size * 2] = pixel[0] / 255.0f; // B

        // 填充深度
        depth_ptr[i] = sparse_depth.at<float>(row, col);
    }

    // ========== 步骤 2: 创建 Tensor ==========
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> rgb_dims = {1, 3, height_, width_};
    std::vector<int64_t> depth_dims = {1, 1, height_, width_};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_rgb_values_.data(), input_rgb_values_.size(), 
        rgb_dims.data(), rgb_dims.size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_depth_values_.data(), input_depth_values_.size(), 
        depth_dims.data(), depth_dims.size()));

    // ========== 步骤 3: 运行推理 ==========
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names_.data(),
            output_node_names_.size()
        );
    } catch (const Ort::Exception& e) {
        std::cerr << "[SGSNet] Inference error: " << e.what() << std::endl;
        return {};
    }

    // ========== 步骤 4: 解析输出 ==========
    // 获取 dense_depth
    float* dense_ptr = output_tensors[0].GetTensorMutableData<float>();
    cv::Mat dense_depth(static_cast<int>(height_), static_cast<int>(width_), CV_32FC1);
    std::memcpy(dense_depth.data, dense_ptr, static_cast<size_t>(image_size) * sizeof(float));

    // 获取 uncertainty (如果存在)
    cv::Mat uncertainty;
    if (output_tensors.size() > 1) {
        float* uncert_ptr = output_tensors[1].GetTensorMutableData<float>();
        uncertainty = cv::Mat(static_cast<int>(height_), static_cast<int>(width_), CV_32FC1);
        std::memcpy(uncertainty.data, uncert_ptr, static_cast<size_t>(image_size) * sizeof(float));
    } else {
        // 如果模型没有输出 uncertainty，创建一个默认的（全零）
        uncertainty = cv::Mat::zeros(static_cast<int>(height_), static_cast<int>(width_), CV_32FC1);
        std::cerr << "[SGSNet] Warning: Model did not output uncertainty, using zeros." << std::endl;
    }

    return {dense_depth, uncertainty};
}
// std::pair<cv::Mat, cv::Mat> SGSNetONNX::infer(const cv::Mat& rgb, const cv::Mat& sparse_depth) {
//     // 检查输入尺寸
//     if (rgb.rows != height_ || rgb.cols != width_) {
//         std::cerr << "[SGSNet] Error: Input size mismatch!" << std::endl;
//         return {};
//     }

//     // === 步骤 1: 数据排布转换 (HWC -> CHW) ===
//     // 这是最关键的一步。OpenCV 是 HWC (BGR)，模型需要 CHW (RGB)
    
//     const int image_size = height_ * width_;
    
//     // 使用指针加速访问
//     float* rgb_ptr = input_rgb_values_.data();
//     float* depth_ptr = input_depth_values_.data();

//     // 遍历像素
//     for (int i = 0; i < image_size; ++i) {
//         int row = i / width_;
//         int col = i % width_;

//         // 获取 RGB 像素 (OpenCV 默认 BGR)
//         cv::Vec3b pixel = rgb.at<cv::Vec3b>(row, col);

//         // 填充到 planar buffer (RRR...GGG...BBB...)
//         // 并进行归一化 (0-255 -> 0.0-1.0)
//         rgb_ptr[i]                = pixel[2] / 255.0f; // R
//         rgb_ptr[i + image_size]   = pixel[1] / 255.0f; // G
//         rgb_ptr[i + image_size*2] = pixel[0] / 255.0f; // B

//         // 填充深度
//         depth_ptr[i] = sparse_depth.at<float>(row, col);
//     }

//     // === 步骤 2: 创建 Tensor ===
//     // 这里的 MemoryInfo 指向 CPU 内存，因为我们刚把数据填到 vector 里
//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

//     std::vector<int64_t> rgb_dims = {1, 3, height_, width_};
//     std::vector<int64_t> depth_dims = {1, 1, height_, width_};

//     std::vector<Ort::Value> input_tensors;
//     input_tensors.push_back(Ort::Value::CreateTensor<float>(
//         memory_info, input_rgb_values_.data(), input_rgb_values_.size(), rgb_dims.data(), rgb_dims.size()));
    
//     input_tensors.push_back(Ort::Value::CreateTensor<float>(
//         memory_info, input_depth_values_.data(), input_depth_values_.size(), depth_dims.data(), depth_dims.size()));

//     // === 步骤 3: 运行推理 ===
//     auto output_tensors = session_->Run(
//         Ort::RunOptions{nullptr},
//         input_node_names_.data(),
//         input_tensors.data(),
//         input_tensors.size(),
//         output_node_names_.data(),
//         output_node_names_.size()
//     );

//     // === 步骤 4: 解析输出 ===
//     // 获取 dense_depth
//     float* dense_ptr = output_tensors[0].GetTensorMutableData<float>();
//     cv::Mat dense_depth(height_, width_, CV_32FC1);
//     // 使用 memcpy 快速复制数据 (注意：如果 TensorRT 做了优化，这里可能是 GPU 指针，但在 ORT C++ API 中，
//     // GetTensorMutableData 默认返回 CPU 可访问的指针，除非显式使用了 IOBinding)
//     std::memcpy(dense_depth.data, dense_ptr, image_size * sizeof(float));

//     // 获取 uncertainty (如果存在)
//     cv::Mat uncertainty;
//     if (output_tensors.size() > 1) {
//         float* uncert_ptr = output_tensors[1].GetTensorMutableData<float>();
//         uncertainty = cv::Mat(height_, width_, CV_32FC1);
//         std::memcpy(uncertainty.data, uncert_ptr, image_size * sizeof(float));
//     }

//     return {dense_depth, uncertainty};
// }