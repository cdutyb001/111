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

#include <vector>

#include <torch/torch.h>

#include <fused-ssim/ssim.h>

namespace loss_utils
{

inline torch::Tensor l1_loss(torch::Tensor &network_output, torch::Tensor &gt)
{
    return torch::abs(network_output - gt).mean();
}

inline torch::Tensor psnr(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).mean();
    return 10.0f * torch::log10(1.0f / mse);
}

/** def psnr(img1, img2):
 *     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
 *     return 20 * torch.log10(1.0 / torch.sqrt(mse))
 */
inline torch::Tensor psnr_gaussian_splatting(torch::Tensor &img1, torch::Tensor &img2)
{
    auto mse = torch::pow(img1 - img2, 2).view({img1.size(0) , -1}).mean(1, /*keepdim=*/true);
    return 20.0f * torch::log10(1.0f / torch::sqrt(mse)).mean();
}

inline torch::Tensor gaussian(
    int window_size,
    float sigma,
    torch::DeviceType device_type = torch::kCUDA)
{
    std::vector<float> gauss_values(window_size);
    for (int x = 0; x < window_size; ++x) {
        int temp = x - window_size / 2;
        gauss_values[x] = std::exp(-temp * temp / (2.0f * sigma * sigma));
    }
    torch::Tensor gauss = torch::tensor(
        gauss_values,
        torch::TensorOptions().device(device_type));
    return gauss / gauss.sum();
}

inline torch::autograd::Variable create_window(
    int window_size,
    int64_t channel,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto _1D_window = gaussian(window_size, 1.5f, device_type).unsqueeze(1);
    auto _2D_window = _1D_window.mm(_1D_window.t()).to(torch::kFloat).unsqueeze(0).unsqueeze(0);
    auto window = torch::autograd::Variable(_2D_window.expand({channel, 1, window_size, window_size}).contiguous());
    return window;
}

/*ssim*/

inline torch::Tensor _ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::autograd::Variable &window,
    int window_size,
    int64_t channel,
    bool size_average = true)
{
    int window_size_half = window_size / 2;
    auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));
    auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel));

    auto mu1_sq = mu1.pow(2);
    auto mu2_sq = mu2.pow(2);
    auto mu1_mu2 = mu1 * mu2;

    auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_sq;
    auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu2_sq;
    auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size_half).groups(channel))
                    - mu1_mu2;

    auto C1 = 0.01 * 0.01;
    auto C2 = 0.03 * 0.03;

    auto ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    if (size_average)
        return ssim_map.mean();
    else
        return ssim_map.mean(1).mean(1).mean(1);
}

inline torch::Tensor ssim(
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::DeviceType device_type = torch::kCUDA,
    int window_size = 11,
    bool size_average = true)
{
    auto channel = img1.size(-3);
    auto window = create_window(window_size, channel, device_type);

    // window = window.to(img1.device());
    window = window.type_as(img1);

    return _ssim(img1, img2, window, window_size, channel, size_average);
}

const float C1 = std::pow(0.01, 2);
const float C2 = std::pow(0.03, 2);

/*fused-ssim*/

class FusedSSIMMap : public torch::autograd::Function<FusedSSIMMap> 
{
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, float C1, float C2, 
                                 torch::Tensor& img1, torch::Tensor& img2) 
    {
        std::string padding = "same";
        bool train = true;

        auto result = fusedssim(C1, C2, img1, img2, train);
        torch::Tensor ssim_map = std::get<0>(result);
        torch::Tensor dm_dmu1 = std::get<1>(result);
        torch::Tensor dm_dsigma1_sq = std::get<2>(result);
        torch::Tensor dm_dsigma12 = std::get<3>(result);

        if (padding == "valid") 
        {
            ssim_map = ssim_map.slice(2, 5, -5).slice(3, 5, -5);
        }

        ctx->save_for_backward({img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
        ctx->saved_data["C1"] = C1;
        ctx->saved_data["C2"] = C2;
        ctx->saved_data["padding"] = padding;

        return ssim_map;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        torch::Tensor img1 = saved[0];
        torch::Tensor img2 = saved[1];
        torch::Tensor dm_dmu1 = saved[2];
        torch::Tensor dm_dsigma1_sq = saved[3];
        torch::Tensor dm_dsigma12 = saved[4];

        float C1 = static_cast<float>(ctx->saved_data["C1"].toDouble());
        float C2 = static_cast<float>(ctx->saved_data["C2"].toDouble());
        std::string padding = ctx->saved_data["padding"].toStringRef();

        torch::Tensor dL_dmap = grad_output[0];
        if (padding == "valid") 
        {
            dL_dmap = torch::zeros_like(img1);
            dL_dmap.slice(2, 5, -5).slice(3, 5, -5) = grad_output[0];
        }

        torch::Tensor grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);

        return {torch::Tensor(), torch::Tensor(), grad, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

inline torch::Tensor fused_ssim(torch::Tensor& img1, torch::Tensor& img2) 
{    
    torch::Tensor map = FusedSSIMMap::apply(C1, C2, img1, img2);
    return map.mean();
}

// ================Probabilistic Gaussian-LIC===================== ↓
/**
     * @brief 不确定性加权深度损失
     * L_depth = mean((1/(U+ε)) * |D_render - D_prior|)
     */
inline torch::Tensor uncertainty_weighted_depth_loss(
    torch::Tensor& rendered_depth,          // 渲染深度（H，W）
    torch::Tensor& prior_depth,             // 先验深度（H，W） from SGSnet
    torch::Tensor& uncertainty_map,         // 不确定性图(H,W)
    float epsilon = 1e-6f
)
{
    // 计算权重：不确定性越低，权重越高
    auto weight = 1.0f/(uncertainty_map + epsilon);

    // L1深度差异
    // 修改前1 ---绝对值差异
    // auto depth_diff = torch::abs(rendered_depth - prior_depth);  
    // 修改后2 --- 相对值差异
    // auto depth_diff = torch::abs(rendered_depth - prior_depth) / (prior_depth + epsilon);
    // 修改后3
    // auto depth_diff = torch::abs(rendered_depth - prior_depth) / prior_depth.clamp_min(1.0f);
    // 修改后4
    auto depth_diff = torch::abs(rendered_depth - prior_depth) / prior_depth.clamp_min(1.0f);

    // 加权损失
    auto weighted_loss = depth_diff * weight;

    // 只计算有效区域（prior_deprh > 0）
    auto valid_mask = (prior_depth > 0.1f) & (rendered_depth > 0.1f);
    auto masked_loss = weighted_loss * valid_mask.to(torch::kFloat32);

    // 归一化
    auto num_valid = valid_mask.sum() + epsilon;
    return masked_loss.sum()/num_valid;
}
/**
     * @brief 从深度图计算法向图
     * 使用Sobel算子计算梯度
     */
inline torch::Tensor depth_to_normal(
    torch::Tensor& depth,   //(H,W)
    float fx,float fy
)
{
    // Sobel核
    auto sobel_x = torch::tensor({{-1.0f,0.0f,1.0f},
                                {-2.0f,0.0f,2.0f},
                                {-1.0f,0.0f,1.0f}}).cuda().unsqueeze(0).unsqueeze(0);
    auto sobel_y = torch::tensor({{-1.0f,-2.0f,-1.0f},
                                {0.0f,0.0f,0.0f},
                                {1.0f,2.0f,1.0f}}).cuda().unsqueeze(0).unsqueeze(0);
    
    // 添加batch和channel维度
    auto depth_4d = depth.unsqueeze(0).unsqueeze(0);

    // 计算梯度
    auto grad_x = torch::nn::functional::conv2d(
        depth_4d,sobel_x,
        torch::nn::functional::Conv2dFuncOptions().padding(1)
    );
    auto grad_y = torch::nn::functional::conv2d(
        depth_4d,sobel_y,
        torch::nn::functional::Conv2dFuncOptions().padding(1)
    );

    //转换为3D法向
    grad_x = grad_x.squeeze()/fx;// dz/dx
    grad_y = grad_y.squeeze()/fy;// dz/dy

    // 法向量 n = （-dz/dx，dz/dy，1） 然后归一下
    auto normal = torch::stack({-grad_x, -grad_y, torch::ones_like(grad_x)}, 0);
    auto normal_norm = torch::norm(normal,2,0,true) + 1e-6F;
    normal = normal / normal_norm;

    return normal; // （3，H，W）
}

 /**
 * @brief 法向一致性损失
 * L_normal = mean((1-U) * (1 - <N_render, N_prior>))
 */
inline torch::Tensor normal_consistency_loss(
    torch::Tensor& rendered_depth,      //渲染深度（H，W）
    torch::Tensor& prior_depth,         //先验深度（H，W）
    torch::Tensor& uncertainty_map,     // 不确定性图（H，W）
    float fx,float fy
)
{
    // 计算两个法向图
    auto normal_render = depth_to_normal(rendered_depth,fx,fy); // (3, H, W)
    auto normal_prior = depth_to_normal(prior_depth,fx,fy); // (3, H, W)

    // 余弦相似度（逐像素点积）
    auto consine_sim = (normal_render * normal_prior).sum(0); //(H,W)

    // 法向误差
    auto normal_error = 1.0f -consine_sim;

    // 置信度加权，不确定性低的区域权重高
    auto confidence = 1.0f - uncertainty_map;
    // auto confidence = torch::clamp(1.0f - uncertainty_map, 0.0f, 1.0f);
    auto weighted_error = normal_error * confidence;

    // 只计算有效区域
    auto valid_mask = (prior_depth > 0.1f) & (rendered_depth > 0.1f);
    auto masked_error = weighted_error * valid_mask.to(torch::kFloat32);

    auto num_valid = valid_mask.sum() + 1e-6f;

    return masked_error.sum()/num_valid;
}

/**
 * @brief 完整的几何约束损失
 */

inline torch::Tensor geometric_constraint_loss(
    torch::Tensor& rendered_depth,
    torch::Tensor& prior_depth,
    torch::Tensor& uncertainty_map,
    float fx,float fy,
    float lambda_depth = 0.1f,
    float lambda_normal = 0.05f
)
{
    auto L_depth = uncertainty_weighted_depth_loss(
        rendered_depth,prior_depth,uncertainty_map);

    auto L_normal = normal_consistency_loss(
        rendered_depth,prior_depth,uncertainty_map,fx,fy);

    return lambda_depth * L_depth + lambda_normal * L_normal;
    
}

inline torch::Tensor pgc_loss(
    torch::Tensor& rendered_depth,          //渲染深度
    torch::Tensor& refined_depth,           //补全深度
    torch::Tensor& uncertainty_map,         //不确定性图
    float threshold = 0.8f,
    float epsilon = 1e-6f
)
{
    // 权重函数：ω(u) = (1/σ) * 𝟙(σ < τ)
    auto weight = torch::where(
        uncertainty_map < threshold,
        1.0f / (uncertainty_map + epsilon),
        torch::zeros_like(uncertainty_map)
    );

    // 加权 L1 损失
    auto diff = torch::abs(rendered_depth - refined_depth);
    return (weight * diff).sum() / weight.sum().clamp_min(1e-6f);
}


// ================Probabilistic Gaussian-LIC===================== ↑

// ================== 混合深度损失 - 分而治之 ==================
// 直接复制以下代码替换 loss_utils.h 第 359-429 行

/**
 * @brief 在线尺度对齐 - 手动最小二乘拟合
 * 避免使用 torch::linalg::lstsq 以解决 LibTorch API 兼容性问题
 */
inline std::pair<float, float> compute_scale_alignment(
    const torch::Tensor& target_depth,
    const torch::Tensor& pred_depth,
    const torch::Tensor& valid_mask
) {
    auto target_d = target_depth.masked_select(valid_mask);
    auto pred_d = pred_depth.masked_select(valid_mask);
    
    int n = target_d.size(0);
    if (n < 10) {
        return {1.0f, 0.0f};
    }
    
    // 手动计算最小二乘：s * pred + t = target
    auto mean_target = target_d.mean();
    auto mean_pred = pred_d.mean();
    
    auto pred_centered = pred_d - mean_pred;
    auto target_centered = target_d - mean_target;
    
    // scale = Σ((target - mean_t) * (pred - mean_p)) / Σ((pred - mean_p)^2)
    auto numerator = (target_centered * pred_centered).sum();
    auto denominator = (pred_centered * pred_centered).sum();
    
    float denom_val = denominator.item<float>();
    if (std::abs(denom_val) < 1e-8f) {
        return {1.0f, 0.0f};
    }
    
    float scale = numerator.item<float>() / denom_val;
    float shift = mean_target.item<float>() - scale * mean_pred.item<float>();
    
    // 物理约束
    scale = std::clamp(scale, 0.1f, 10.0f);
    
    return {scale, shift};
}

/**
 * @brief 混合深度损失 - 分而治之策略
 */
inline torch::Tensor hybrid_depth_loss(
    torch::Tensor& render_depth,
    torch::Tensor& sgsnet_depth,
    torch::Tensor& sgsnet_uncertainty,
    torch::Tensor& sparse_depth,
    float lambda_lidar = 1.0f,
    float lambda_sgs = 0.1f,
    bool use_scale_align = true,
    bool print_debug = false
) {
    auto valid_mask = (sparse_depth > 0.1f).detach();
    int num_lidar_pts = valid_mask.sum().item<int>();
    
    float scale = 1.0f, shift = 0.0f;
    torch::Tensor aligned_sgs_depth;
    
    if (use_scale_align && num_lidar_pts > 10) {
        std::tie(scale, shift) = compute_scale_alignment(
            sparse_depth, sgsnet_depth, valid_mask
        );
        aligned_sgs_depth = (sgsnet_depth * scale + shift).detach();
    } else {
        aligned_sgs_depth = sgsnet_depth.detach();
    }
    
    // LiDAR 强约束
    auto loss_lidar = torch::abs(render_depth - sparse_depth) * 
                      valid_mask.to(torch::kFloat32);
    
    // SGSNet 软约束
    auto sgs_weight = (1.0f - sgsnet_uncertainty).clamp(0.0f, 1.0f).detach();
    auto hole_mask = (~valid_mask).to(torch::kFloat32);
    auto loss_sgs = torch::abs(render_depth - aligned_sgs_depth) * 
                    hole_mask * sgs_weight;
    
    float total_pixels = static_cast<float>(render_depth.numel()) + 1e-6f;
    auto lidar_contrib = loss_lidar.sum() * lambda_lidar / total_pixels;
    auto sgs_contrib = loss_sgs.sum() * lambda_sgs / total_pixels;
    auto loss = lidar_contrib + sgs_contrib;
    
    if (print_debug) {
        std::cout << "[Hybrid Depth] "
                  << "LiDAR=" << num_lidar_pts 
                  << " Scale=" << scale 
                  << " | L_lidar=" << lidar_contrib.item<float>()
                  << " L_sgs=" << sgs_contrib.item<float>()
                  << " Total=" << loss.item<float>() << std::endl;
    }
    
    return loss;
}

/**
 * @brief 完整的混合几何约束损失
 */
inline torch::Tensor hybrid_geometric_constraint_loss(
    torch::Tensor& render_depth,
    torch::Tensor& sgsnet_depth,
    torch::Tensor& sgsnet_uncertainty,
    torch::Tensor& sparse_depth,
    float fx, float fy,
    float lambda_lidar = 1.0f,
    float lambda_sgs = 0.1f,
    float lambda_normal = 0.05f,
    bool use_scale_align = true
) {
    auto L_depth = hybrid_depth_loss(
        render_depth, sgsnet_depth, sgsnet_uncertainty, sparse_depth,
        lambda_lidar, lambda_sgs, use_scale_align, false
    );
    
    auto valid_mask = (sparse_depth > 0.1f).detach();
    float scale = 1.0f, shift = 0.0f;
    if (use_scale_align) {
        std::tie(scale, shift) = compute_scale_alignment(
            sparse_depth, sgsnet_depth, valid_mask
        );
    }
    auto aligned_sgs = (sgsnet_depth * scale + shift).detach();
    
    auto L_normal = normal_consistency_loss(
        render_depth, aligned_sgs, sgsnet_uncertainty, fx, fy
    );
    
    return L_depth + lambda_normal * L_normal;
}

// ================== 混合深度损失 - 结束 ==================

}
