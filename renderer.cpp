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

#include "renderer.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, 
            torch::Tensor, torch::Tensor, torch::Tensor>
render(const std::shared_ptr<Camera>& viewpoint_camera,
       std::shared_ptr<GaussianModel> pc,
       torch::Tensor& bg_color,
       bool use_trained_exposure,
       bool no_color,
       float scaling_modifier)
{
    auto screenspace_points = torch::zeros_like(pc->getXYZ(), torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));

    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);  // w / (2 * fx)
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);  // h / (2 * fy)
    bool prefiltered = false;
    bool debug = false;
    // ================Probabilistic Gaussian-LIC===================== ↓
    // 获取mip=splatting 和相关参数
    float mip_filter_var = pc->mip_filter_var_;
    float uncertainty_beta = pc->uncertainty_beta_;
    bool enable_probabilistic = pc->enable_probabilistic_;
    // ================Probabilistic Gaussian-LIC===================== ↑
    GaussianRasterizationSettings raster_settings(
        viewpoint_camera->image_height_,
        viewpoint_camera->image_width_,
        tanfovx,
        tanfovy,
        viewpoint_camera->limx_neg_,
        viewpoint_camera->limx_pos_,
        viewpoint_camera->limy_neg_,
        viewpoint_camera->limy_pos_,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->sh_degree_,
        viewpoint_camera->camera_center_,
        prefiltered,
        debug,
        no_color,
        pc->lambda_erank_,
        // pc->mip_filter_var_  // ← 新增参数
        // ================Probabilistic Gaussian-LIC===================== ↓ 
        mip_filter_var,
        uncertainty_beta,
        enable_probabilistic
        // ================Probabilistic Gaussian-LIC===================== ↑
    );
    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();  // (n, 3)
    auto means2D = screenspace_points;  // (n, 3)
    auto opacity = pc->getOpacity();  // (n, 1) 0-1
    // # ================Probabilistic Gaussian-LIC===================== version 2
    auto scales = enable_probabilistic ? 
                  pc->getUncertaintyModulatedScaling() : 
                  pc->getScaling();
    // # ================Probabilistic Gaussian-LIC===================== version 2
    auto rotations = pc->getRotation();  // (n, 4)
    torch::Tensor dc = pc->getFeaturesDc();  // (n, 1, 3)
    torch::Tensor shs = pc->getFeaturesRest();  // (n, 15, 3)
    torch::Tensor colors_precomp; 
    torch::Tensor cov3D_precomp;

    auto rasterizer_result = rasterizer.forward(
                                    means3D,
                                    means2D,
                                    opacity,
                                    dc,
                                    shs,
                                    colors_precomp,
                                    scales,
                                    rotations,
                                    cov3D_precomp);
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);
    auto rendered_final_T = std::get<2>(rasterizer_result);
    auto rendered_depth = std::get<3>(rasterizer_result);  // ← 新增

    return std::make_tuple(
        rendered_image,     // RGB 图像
        rendered_final_T,   // 透射率 T
        rendered_depth,     // 渲染深度 ← 新增
        screenspace_points, 
        radii > 0,          
        radii
    );
}