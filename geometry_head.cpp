#include "geometry_head.h"

#include"tensor_utils.h"
#include<omp.h>
#include<opencv2/ximgproc.hpp>
#include<SGSNet/sgsnet_onnx.h> // SGSNet补全

// GeometryHead::GeometryHead(int height,int width,float fx,float fy,float cx,float cy):height_(height),width_(width),fx_(fx),fy_(fy),cx_(cx),cy_(cy){}
// ================Probabilistic Gaussian-LIC===================== version 2
GeometryHead::GeometryHead(int height, int width, float fx, float fy, float cx, float cy)
            : height_(height), width_(width), fx_(fx), fy_(fy), cx_(cx), cy_(cy), use_sgsnet_(false)
{
    std::cout << "[GeometryHead] Initialized without SGSNet (bilateral filter mode)." << std::endl;
}
GeometryHead::GeometryHead(int height,int width,float fx,float fy,float cx,float cy,const std::string& sgsnet_model_path,bool use_cuda)
            :height_(height),width_(width),fx_(fx),fy_(fy),cx_(cx),cy_(cy),use_sgsnet_(true)
{
    try{
        sgsnet_engine_ = std::make_unique<SGSNetONNX>(sgsnet_model_path, height, width, use_cuda);
        std::cout << "[GeometryHead] SGSNet initialized with resolution " 
                  << width << "x" << height << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr<<"[GeometryHead] SGSNet init failed: " <<e.what()<<std::endl;
        std::cerr<<"[GeometryHead] Falling back to bilateral filtering."<<std::endl;
        sgsnet_engine_ = nullptr;
        use_sgsnet_ = false;
    }
}

// ================Probabilistic Gaussian-LIC===================== version 2
// ============================================================================
// 运行时切换 SGSNet/双边滤波
// ============================================================================
void GeometryHead::setUseSGSNet(bool ust){
    if(ust && sgsnet_engine_ == nullptr){
        std::cerr << "[GeometryHead] Warning: SGSNet engine not initialized, cannot enable." << std::endl;
        return;
    }
    use_sgsnet_ = ust;
    std::cout << "[GeometryHead] Switched to " << (ust ? "SGSNet" : "bilateral filter") << " mode." << std::endl;
}

// ============================================================================
// 投影 LiDAR 点云到图像平面，生成稀疏深度图
// ============================================================================
cv::Mat GeometryHead::projectLidarToImage(
    const std::vector<Eigen::Vector3d>& lidar_points,
    const Eigen::Matrix3d& R_cw,
    const Eigen::Vector3d& t_cw
)
{
    cv::Mat sparse_depth = cv::Mat::zeros(height_,width_,CV_32FC1);
    cv::Mat depth_count = cv::Mat::zeros(height_,width_,CV_32FC1);

    for(const auto& p_w:lidar_points)
    {
        // 世界坐标->相机坐标
        Eigen::Vector3d p_c = R_cw*p_w+t_cw;

        // 检查是否在相机的前方
        if(p_c.z()<=0.1) continue;

        //投影到图像平面
        float u = fx_*p_c.x()/p_c.z()+cx_;
        float v = fy_*p_c.y()/p_c.z()+cy_;

        int iu = static_cast<int>(std::round(u));
        int iv = static_cast<int>(std::round(v));

        // 检查边界
        if(iu < 0 || iu >= width_ || iv < 0 || iv >= height_) continue;

        float depth = static_cast<float>(p_c.z());

        // 如果该位还没有更新深度、或者新深度更新更近，则更新
        if(sparse_depth.at<float>(iv,iu) == 0 || depth < sparse_depth.at<float>(iv,iu))
        {
            sparse_depth.at<float>(iv,iu) = depth;
        }
    }
    return sparse_depth;

}
// ============================================================================
// 双边滤波稠密化（Fallback 方法）
// ============================================================================
cv::Mat GeometryHead::bilateraDensification(
    const cv::Mat& sparse_depth,
    const cv::Mat& rgb_image
)
{
    // // 创建有效深度掩码
    // const cv::Mat& valid_mask = sparse_depth>0;
    // valid_mask.convertTo(valid_mask,CV_32FC1,1.0/255.0);

    // 初始化稠密深度
    cv::Mat dense_depth = sparse_depth.clone();
    // 迭代双边滤波填充
    for(int iter = 0;iter<filter_interations_;iter++)
    {
        cv::Mat filtered;

        // 使用双边联合滤波（以RGB图像为引导）
        cv::Mat rgb_float;
        rgb_image.convertTo(rgb_float,CV_32FC3);

        // Opencv 的jointBilateralFilter
        cv::ximgproc::jointBilateralFilter(
            rgb_float,                  // 引导图像
            dense_depth,                // 输入
            filtered,                   // 输出
            -1,                         // d = -1 根据 sigmaSpace计算
            sigma_color_,               // sigmaColor
            sigma_space_                // sigmaSpace
        );

        // 只更新无效位置（保持原始Lidar深度）
        for(int y = 0;y<height_;y++)
            for(int x = 0;x<width_;x++)
                if(sparse_depth.at<float>(y,x) == 0)
                    dense_depth.at<float>(y,x) = filtered.at<float>(y,x);

    }


    return dense_depth;
}

// ============================================================================
// 计算不确定性图（用于双边滤波 fallback）
// ============================================================================

cv::Mat GeometryHead::computeUncertainty(
    const cv::Mat& sparse_mask,
    const cv::Mat& rgb_image
)
{
    cv::Mat uncertainty = cv::Mat::ones(height_,width_,CV_32FC1);

    // 距离变换，计算每个像素到最近Lidar点的距离
    cv::Mat binary_mask;
    sparse_mask.convertTo(binary_mask,CV_8UC1,255);

    cv::Mat dist;
    cv::distanceTransform(255-binary_mask,dist,cv::DIST_L2,5);

    //归一化距离到【0，1】
    double maxDist;
    cv::minMaxLoc(dist,nullptr,&maxDist);
    if(maxDist>0){
        dist/=maxDist;
    }

    // 基于距离的不确定性基础值
    cv::Mat base_uncertainty = dist.clone();

    // 添加基于纹理的调制（弱纹理区域不确定性更高）
    cv::Mat gray;
    if (rgb_image.channels() == 3)
    {
        cv::cvtColor(rgb_image,gray,cv::COLOR_RGB2GRAY);
    }else{
        gray = rgb_image.clone();
    }
    gray.convertTo(gray,CV_32FC1);


    // 计算局部方差作为纹理度量
    cv::Mat local_mean,local_sqmean;
    cv::blur(gray,local_mean,cv::Size(11,11));
    cv::blur(gray.mul(gray),local_sqmean,cv::Size(11,11));
    cv::Mat variance = local_sqmean - local_mean(local_mean);


    // 归一化方差
    double maxVar;
    cv::minMaxLoc(variance,nullptr,&maxVar);
    if(maxVar > 0){
        variance/=maxVar;
    }

    // 弱纹理 = 低方差 = 高不确定性
    cv::Mat texture_uncertainty = 1.0 - variance;

    //组合不确定性：距离+纹理
    uncertainty = 0.6f*base_uncertainty+0.4f*texture_uncertainty;

    //裁剪到【0，1】
    cv::threshold(uncertainty,uncertainty,1.0,1.0,cv::THRESH_TRUNC);
    cv::threshold(uncertainty,uncertainty,0.0,0.0,cv::THRESH_TOZERO);
    
    // 原始LiDar点位置不确定性为0
    for (int y = 0; y < height_; y++)
    {
        
        for (int x = 0; x <width_; x++)
        {
            if (sparse_mask.at<float>(y,x)>0)
            {
                uncertainty.at<float>(y,x) = 0.0f;
            }
            
        }
        
    }

    return uncertainty;
    
}

// ================Probabilistic Gaussian-LIC===================== version 2↓
// ============================================================================
// SGSNet 深度补全方法
// ============================================================================
std::pair<cv::Mat,cv::Mat> GeometryHead::sgsnetDensification(
    const cv::Mat& sparse_depth,
    const cv::Mat& rgb_image
)
{
    // step1 格式转换（确保 CV_8UC3）
    cv::Mat rgb_input;
    if(rgb_image.type() == CV_8UC3){
        rgb_input = rgb_image;
    }
    else if(rgb_image.type() == CV_32FC3){
        rgb_image.convertTo(rgb_input,CV_8UC3,255.0);
    }
    else {
        std::cerr << "[GeometryHead] Warning: Unexpected RGB type " << rgb_image.type() 
                  << ", attempting conversion." << std::endl;
        rgb_image.convertTo(rgb_input,CV_8UC3,255.0);
    }

    // // step2 调整尺寸（SGSNet 期望 480*640）
    // const int target_h = 480,target_w = 640;
    // cv::Mat rgb_resized,depth_resized;
    // bool need_resize = (rgb_input.rows != target_h || rgb_input.cols != target_w);
    // if(need_resize){
    //     cv::resize(rgb_input,rgb_resized,cv::Size(target_w,target_h));
    //     cv::resize(sparse_depth,depth_resized,cv::Size(target_w,target_h),0,0,cv::INTER_NEAREST);
    // }
    // else{
    //     // ✅ 添加 else 分支
    //     rgb_resized = rgb_input;
    //     depth_resized = sparse_depth;
    // }

    // 确保深度图是 CV_32FC1
    cv::Mat depth_input;
    if(sparse_depth.type() != CV_32FC1)
    {
        sparse_depth.convertTo(depth_input,CV_32FC1);
    }
    else{
        depth_input = sparse_depth;
    }

    // step3 执行推理
    auto [dense_depth,uncertainty] = sgsnet_engine_->infer(rgb_input,depth_input);

    // // step4 恢复原始尺寸
    // if(need_resize){
    //     cv::resize(dense_depth,dense_depth,cv::Size(width_,height_));
    //     cv::resize(uncertainty,uncertainty ,cv::Size(width_,height_));
    // }
    // 检查推理结果
    // 检查推理结果
    if (dense_depth.empty()) {
        std::cerr << "[GeometryHead] SGSNet inference failed, falling back to bilateral filter." << std::endl;
        cv::Mat fallback_depth = bilateraDensification(sparse_depth, rgb_image);
        cv::Mat sparse_mask = (sparse_depth > 0);
        sparse_mask.convertTo(sparse_mask, CV_32FC1, 1.0 / 255.0);
        cv::Mat fallback_uncertainty = computeUncertainty(sparse_mask, rgb_image);
        return {fallback_depth, fallback_uncertainty};
    }
    // ========== Step 4: 不再需要 resize 回原始尺寸！==========
    // 输出已经是原始分辨率

    return {dense_depth,uncertainty};
}
// ============================================================================
// 主处理函数
// ============================================================================
GeometryOutput GeometryHead::process(
    const cv::Mat& image,
    const std::vector<Eigen::Vector3d>& lidar_points,
    const Eigen::Matrix3d& R_cw,
    const Eigen::Vector3d& t_cw
)
{
    GeometryOutput output;

    // step1 投影Lidar生成稀疏深度图
    cv::Mat sparse_depth = projectLidarToImage(lidar_points,R_cw,t_cw);

    // ⬅️ 新增：保存稀疏深度到输出
    output.sparse_depth_cv = sparse_depth.clone();
    output.sparse_depth = torch::from_blob(
        sparse_depth.data, 
        {height_, width_}, 
        torch::kFloat32
    ).clone();  // clone() 确保数据独立

    cv::Mat sparse_mask = (sparse_depth>0);
    sparse_mask.convertTo(sparse_mask,CV_32FC1,1.0/255.0);

    // step2 双边滤波稠密化 二选一
    // output.dense_depth_cv = bilateraDensification(sparse_depth,image);
    // step 2 深度补全 - SGSNet / 原始双边滤波
    if(isUsingSGSNet() && sgsnet_engine_ != nullptr)
    {
        // 使用SGSNet
        auto [dense_depth,uncertainty] = sgsnetDensification(sparse_depth,image);
        output.dense_depth_cv = dense_depth;
        output.uncertainty_cv = uncertainty;
        std::cout<<"使用了 《SGSNet》 深度补全 ("<< dense_depth.cols << "x" << dense_depth.rows << ")" << std::endl;
    }
    else
    {
        // 双边滤波
        output.dense_depth_cv = bilateraDensification(sparse_depth, image);
        output.uncertainty_cv = computeUncertainty(sparse_mask, image);
        std::cout<<"使用了 《双边滤波》 深度补全 "<<std::endl;
    }

    // ========================================================================
    // step2.5 ✅ 新增：清理和验证深度图（防止 CUDA 非法内存访问）
    // ========================================================================

    const float MIN_DEPTH = 0.1f;       // 最小深度 10cm
    const float MAX_DEPTH= 200.0f;     // 最大深度 100m

    cv::Mat& depth = output.dense_depth_cv;

    //1 处理NaN
    cv::patchNaNs(depth,0.0f);
    //2 处理Inf（Ing不等于Inf*0.5）
    cv::Mat inf_mask = (depth == depth + 1.0f) & (depth != 0.0f);
    depth.setTo(0.0f , inf_mask);
    //3 限制深度范围
    cv::Mat too_close = (depth > 0) & (depth < MIN_DEPTH);
    cv::Mat too_fat = (depth > MAX_DEPTH);
    depth.setTo(0.0f,too_close);
    depth.setTo(MAX_DEPTH,too_fat);
    //4 统计无效像素
    int total_pixels = depth.rows * depth.cols;
    int invalid_count = cv::countNonZero(inf_mask | too_close | too_fat);

    if(invalid_count > 0){
        double invalid_ratio = 100.0 * invalid_count / total_pixels;
        std::cout<< "\033[1;33m[GeometryHead] Cleaned " <<invalid_count
                 << " invalid pixels ("<< std::fixed << std::setprecision(2)
                 << invalid_ratio << "%)\033[0m" << std::endl;
    }
    // 5 输出深度范围统计
    double minVal,maxVal;
    cv::minMaxLoc(depth , &minVal,&maxVal);
    if(maxVal > 0){
        std::cout << "\033[1;32m[GeometryHead] Depth range: [" 
                      << std::fixed << std::setprecision(2) << minVal << ", " 
                      << maxVal << "] m\033[0m" << std::endl;
    }
    //6 清理不确定性图
    if(!output.uncertainty_cv.empty()){
        cv::patchNaNs(output.uncertainty_cv , 1.0f);
        // 限制到【0，1】范围
        output.uncertainty_cv = cv::max(0.0f , cv::min(output.uncertainty_cv,1.0f));
    }


    // step3 转换为Tensor
    output.dense_depth = tensor_utils::cvMat2TorchTensor_Float32(
        output.dense_depth_cv,torch::kCUDA
    );
    output.uncertainty_map = tensor_utils::cvMat2TorchTensor_Float32(
        output.uncertainty_cv,torch::kCUDA
    );
    output.sparse_depth = tensor_utils::cvMat2TorchTensor_Float32(
    output.sparse_depth_cv, torch::kCUDA
);

    return output;

}
// ================Probabilistic Gaussian-LIC===================== version 2↑

// #include "geometry_head.h"
// #include "tensor_utils.h"
// #include <omp.h>
// #include <opencv2/ximgproc.hpp>

// GeometryHead::GeometryHead(int height, int width, float fx, float fy, float cx, float cy)
//     : height_(height), width_(width), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

// cv::Mat GeometryHead::projectLidarToImage(
//     const std::vector<Eigen::Vector3d>& lidar_points,
//     const Eigen::Matrix3d& R_cw,
//     const Eigen::Vector3d& t_cw)
// {
//     cv::Mat sparse_depth = cv::Mat::zeros(height_, width_, CV_32FC1);

//     for (const auto& p_w : lidar_points)
//     {
//         // 世界坐标->相机坐标
//         Eigen::Vector3d p_c = R_cw * p_w + t_cw;

//         // 检查是否在相机的前方
//         if (p_c.z() <= 0.1) continue;

//         // 投影到图像平面
//         float u = fx_ * p_c.x() / p_c.z() + cx_;
//         float v = fy_ * p_c.y() / p_c.z() + cy_;

//         int iu = static_cast<int>(std::round(u));
//         int iv = static_cast<int>(std::round(v));

//         // 检查边界
//         if (iu < 0 || iu >= width_ || iv < 0 || iv >= height_) continue;

//         float depth = static_cast<float>(p_c.z());

//         // 如果该位还没有更新深度、或者新深度更近，则更新
//         if (sparse_depth.at<float>(iv, iu) == 0 || depth < sparse_depth.at<float>(iv, iu))
//         {
//             sparse_depth.at<float>(iv, iu) = depth;
//         }
//     }  // for 循环结束

//     return sparse_depth;  // ✅ 移到 for 循环外部
// }

// cv::Mat GeometryHead::bilateraDensification(
//     const cv::Mat& sparse_depth,
//     const cv::Mat& rgb_image)
// {
//     // 初始化稠密深度
//     cv::Mat dense_depth = sparse_depth.clone();
    
//     // 创建有效深度掩码（有深度值的位置）
//     cv::Mat valid_mask = (sparse_depth > 0);
    
//     // 方法1：使用 OpenCV inpaint 填充空洞
//     cv::Mat mask_inv;
//     cv::bitwise_not(valid_mask, mask_inv);  // 反转掩码，标记需要填充的区域
    
//     // 将深度图转换为 8 位用于 inpaint（先归一化）
//     double minVal, maxVal;
//     cv::minMaxLoc(sparse_depth, &minVal, &maxVal, nullptr, nullptr, valid_mask);
    
//     if (maxVal <= minVal) {
//         // 没有有效深度值，返回零图
//         return cv::Mat::zeros(height_, width_, CV_32FC1);
//     }
    
//     // 归一化到 0-255
//     cv::Mat depth_normalized;
//     sparse_depth.convertTo(depth_normalized, CV_8UC1, 255.0 / maxVal);
    
//     // 使用 inpaint 填充空洞
//     cv::Mat depth_inpainted;
//     cv::inpaint(depth_normalized, mask_inv, depth_inpainted, 5, cv::INPAINT_TELEA);
    
//     // 转换回原始范围
//     depth_inpainted.convertTo(dense_depth, CV_32FC1, maxVal / 255.0);
    
//     // 方法2：使用引导滤波平滑（可选，让深度边缘与 RGB 边缘对齐）
//     cv::Mat gray;
//     if (rgb_image.channels() == 3) {
//         cv::cvtColor(rgb_image, gray, cv::COLOR_BGR2GRAY);
//     } else {
//         gray = rgb_image.clone();
//     }
//     gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
    
//     // 引导滤波
//     cv::Mat guided_depth;
//     cv::ximgproc::guidedFilter(gray, dense_depth, guided_depth, 8, 0.01);
    
//     // 保持原始 LiDAR 点的深度值不变
//     for (int y = 0; y < height_; y++) {
//         for (int x = 0; x < width_; x++) {
//             if (sparse_depth.at<float>(y, x) > 0) {
//                 guided_depth.at<float>(y, x) = sparse_depth.at<float>(y, x);
//             }
//         }
//     }
    
//     return guided_depth;
// }

// cv::Mat GeometryHead::computeUncertainty(
//     const cv::Mat& sparse_mask,
//     const cv::Mat& rgb_image)
// {
//     cv::Mat uncertainty = cv::Mat::ones(height_, width_, CV_32FC1);

//     // 距离变换，计算每个像素到最近LiDAR点的距离
//     cv::Mat binary_mask;
//     sparse_mask.convertTo(binary_mask, CV_8UC1, 255);

//     cv::Mat dist;
//     cv::distanceTransform(255 - binary_mask, dist, cv::DIST_L2, 5);

//     // 归一化距离到 [0, 1]
//     double maxDist;
//     cv::minMaxLoc(dist, nullptr, &maxDist);
//     if (maxDist > 0)
//     {
//         dist /= maxDist;
//     }

//     // 基于距离的不确定性基础值
//     cv::Mat base_uncertainty = dist.clone();

//     // 添加基于纹理的调制（弱纹理区域不确定性更高）
//     cv::Mat gray;
//     if (rgb_image.channels() == 3)
//     {
//         cv::cvtColor(rgb_image, gray, cv::COLOR_BGR2GRAY);  // ✅ 改为 BGR2GRAY
//     }
//     else
//     {
//         gray = rgb_image.clone();
//     }
//     gray.convertTo(gray, CV_32FC1);

//     // 计算局部方差作为纹理度量
//     cv::Mat local_mean, local_sqmean;
//     cv::blur(gray, local_mean, cv::Size(11, 11));
//     cv::blur(gray.mul(gray), local_sqmean, cv::Size(11, 11));
//     cv::Mat variance = local_sqmean - local_mean.mul(local_mean);  // ✅ 修复语法错误

//     // 归一化方差
//     double maxVar;
//     cv::minMaxLoc(variance, nullptr, &maxVar);
//     if (maxVar > 0)
//     {
//         variance /= maxVar;
//     }

//     // 弱纹理 = 低方差 = 高不确定性
//     cv::Mat texture_uncertainty = 1.0 - variance;

//     // 组合不确定性：距离 + 纹理
//     uncertainty = 0.6f * base_uncertainty + 0.4f * texture_uncertainty;

//     // 裁剪到 [0, 1]
//     cv::threshold(uncertainty, uncertainty, 1.0, 1.0, cv::THRESH_TRUNC);
//     cv::threshold(uncertainty, uncertainty, 0.0, 0.0, cv::THRESH_TOZERO);

//     // 原始LiDAR点位置不确定性为0
//     for (int y = 0; y < height_; y++)
//     {
//         for (int x = 0; x < width_; x++)
//         {
//             if (sparse_mask.at<float>(y, x) > 0)
//             {
//                 uncertainty.at<float>(y, x) = 0.0f;
//             }
//         }
//     }

//     return uncertainty;
// }

// GeometryOutput GeometryHead::process(
//     const cv::Mat& image,
//     const std::vector<Eigen::Vector3d>& lidar_points,
//     const Eigen::Matrix3d& R_cw,
//     const Eigen::Vector3d& t_cw)
// {
//     GeometryOutput output;

//     // step1 投影LiDAR生成稀疏深度图
//     cv::Mat sparse_depth = projectLidarToImage(lidar_points, R_cw, t_cw);
//     cv::Mat sparse_mask = (sparse_depth > 0);
//     sparse_mask.convertTo(sparse_mask, CV_32FC1, 1.0 / 255.0);

//     // step2 双边滤波稠密化
//     output.dense_depth_cv = bilateraDensification(sparse_depth, image);

//     // step3 计算不确定性图
//     output.uncertainty_cv = computeUncertainty(sparse_mask, image);

//     // step4 转换为Tensor
//     output.dense_depth = tensor_utils::cvMat2TorchTensor_Float32(
//         output.dense_depth_cv, torch::kCUDA);
//     output.uncertainty_map = tensor_utils::cvMat2TorchTensor_Float32(
//         output.uncertainty_cv, torch::kCUDA);

//     return output;  // ✅ 添加返回语句
// }