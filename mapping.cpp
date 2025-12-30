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

#include "mapping.h"
#include "gaussian.h"
#include "geometry_head.h"
#include<SGSNet/sgsnet_onnx.h>

//// ✅ 新增: PCL 和 cv_bridge
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>


std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);

// [新增] 辅助函数：将 3D 点云投影为 2D 稀疏深度图   version 2
cv::Mat projectCloudToDepth(const std::vector<Eigen::Vector3d>& points, 
                            int height, int width, 
                            const Params& prm) {
    cv::Mat sparse_depth = cv::Mat::zeros(height, width, CV_32FC1);
    
    for (const auto& p : points) {
        // 假设 points 已经是相机坐标系下的点 (如果不是，需要先乘外参 T_cw)
        // 通常 /points_for_gs 传来的如果是 lidar frame，这里需要外参
        // 这里假设您的 GeometryHead 内部处理逻辑也是类似的投影
        
        if (p.z() <= 0.1) continue; // 过滤掉无效深度

        // 投影公式: u = fx * x / z + cx, v = fy * y / z + cy
        int u = std::round(prm.fx * p.x() / p.z() + prm.cx);
        int v = std::round(prm.fy * p.y() / p.z() + prm.cy);

        if (u >= 0 && u < width && v >= 0 && v < height) {
            // 获取该位置当前的深度
            float current_d = sparse_depth.at<float>(v, u);
            // 如果该位置没有点，或者新的点更近（遮挡关系），则更新
            if (current_d == 0.0f || p.z() < current_d) {
                sparse_depth.at<float>(v, u) = static_cast<float>(p.z());
            }
        }
    }
    return sparse_depth;
}

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg) 
{
    m_buf.lock();
    point_buf.push(point_msg);
    last_point_time = ros::Time::now().toSec();
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) 
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg) 
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty()) 
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();

    while (1) 
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();

    return true;
}

void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);

    // ==================== [新增] 初始化 GeometryHead ====================
    // 从 YAML 配置读取 SGSNet 路径
    // [新增] 在进入主循环前，初始化推理引擎 (加载 ONNX 模型)
    // 这样避免了每帧都重新加载模型，大幅提升速度
    // std::string sgsnet_model_path = "/root/catkin_gaussian/src/Gaussian_lic/src/SGSNet/models/sgsnet.onnx";
    // if (node["sgsnet"] && node["sgsnet"]["model_path"]) {
    //     sgsnet_model_path = node["sgsnet"]["model_path"].as<std::string>();
    // }
    // bool use_sgsnet = true;
    // if (node["sgsnet"] && node["sgsnet"]["enabled"]) {
    //     use_sgsnet = node["sgsnet"]["enabled"].as<bool>();
    // }

    // // 创建 GeometryHead 实例
    // std::unique_ptr<GeometryHead> geo_head;
    // if (use_sgsnet) {
    //     geo_head = std::make_unique<GeometryHead>(
    //         prm.height, prm.width, 
    //         prm.fx, prm.fy, prm.cx, prm.cy,
    //         sgsnet_model_path, true  // use_cuda = true
    //     );
    // } else {
    //     geo_head = std::make_unique<GeometryHead>(
    //         prm.height, prm.width, 
    //         prm.fx, prm.fy, prm.cx, prm.cy
    //     );
    // }
    // // ==================== [新增] 结束 ====================

    // std::chrono::steady_clock::time_point t_start, t_end;
    // double total_mapping_time = 0;
    // double total_adding_time = 0;
    // double total_extending_time = 0;
    // //================================================================version2
    // // 创建GeometryHead 用于深度补全（仅当启用概率模式时）
    // std::shared_ptr<GeometryHead> geometry_head = nullptr;
    // bool enable_probabilistic = false;                          //  TODOL: 从prm 读取

    // // 从 YAML 读取配置（需要先修改Params类）
    // if(node["enable_probabilistic"])
    // {
    //     enable_probabilistic = node["enable_probabilistic"].as<bool>();
    // }

    // if(enable_probabilistic)
    // {
    //     geometry_head = std::make_shared<GeometryHead>(
    //         prm.height,prm.width,
    //         static_cast<float>(prm.fx),static_cast<float>(prm.fy),
    //         static_cast<float>(prm.cx),static_cast<float>(prm.cy)
    //     );

    //     std::cout<<"\n"<<std::endl;
    //     std::cout << "\033[1;35m[Mapping] Probabilistic Gaussian-LIC enabled\033[0m" << std::endl;
    // }

    // // 可选部分：创建SGSNet推理器
    // std::shared_ptr<SGSNetONNX> sgsnet = nullptr;
    // if(enable_probabilistic){
    //     sgsnet = std::make_shared<SGSNetONNX>("models/sgsnet.onnx", prm.height, prm.width, true);
    // }
    bool enable_probabilistic = false;
    std::string sgsnet_model_path = "/root/catkin_gaussian/src/Gaussian_lic/src/SGSNet/models/sgsnet.onnx";
    bool use_sgsnet = true;
    
    if(node["enable_probabilistic"]) {
        enable_probabilistic = node["enable_probabilistic"].as<bool>();
    }
    if (node["sgsnet"] && node["sgsnet"]["model_path"]) {
        sgsnet_model_path = node["sgsnet"]["model_path"].as<std::string>();
    }
    if (node["sgsnet"] && node["sgsnet"]["enabled"]) {
        use_sgsnet = node["sgsnet"]["enabled"].as<bool>();
    }
    
    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;
    
    // 创建统一的 GeometryHead 实例
    std::shared_ptr<GeometryHead> geometry_head = nullptr;
    
    if(enable_probabilistic) {
        std::cout << "\033[1;35m[Mapping] Probabilistic Gaussian-LIC enabled\033[0m" << std::endl;
        
        if(use_sgsnet) {
            // 使用 SGSNet 深度补全（集成在 GeometryHead 内部）
            geometry_head = std::make_shared<GeometryHead>(
                prm.height, prm.width,
                static_cast<float>(prm.fx), static_cast<float>(prm.fy),
                static_cast<float>(prm.cx), static_cast<float>(prm.cy),
                sgsnet_model_path, true  // 启用 SGSNet + CUDA
            );
        } else {
            // 使用双边滤波
            geometry_head = std::make_shared<GeometryHead>(
                prm.height, prm.width,
                static_cast<float>(prm.fx), static_cast<float>(prm.fy),
                static_cast<float>(prm.cx), static_cast<float>(prm.cy)
            );
        }
    }
    //================================================================version2

    Frame cur_frame;
    while (!exit_flag)
    {
        /// [1] data alignment
        m_buf.lock();
        bool align_flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!align_flag) continue;
        
        /// [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
       
        if (dataset->is_keyframe_current_)
        {
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        }
        else continue;
        
        // ==================== [新增] 深度补全处理 ====================
        /// [2.5] 深度补全（Probabilistic Gaussian-LIC）
        GeometryOutput geo_output;
        if (enable_probabilistic && geometry_head) 
        {
            // 从 ROS 消息中提取图像
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
                cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image_bgr = cv_ptr->image;
            cv::Mat image_rgb;
            cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
            
            // 提取位姿
            Eigen::Quaterniond q_wc;
            Eigen::Vector3d t_wc;
            tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
            tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
            Eigen::Matrix3d R_wc = q_wc.toRotationMatrix();
            
            // 计算世界到相机的变换
            Eigen::Matrix3d R_cw = R_wc.transpose();
            Eigen::Vector3d t_cw = -R_cw * t_wc;
            
            // 提取 LiDAR 点云
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
            std::vector<Eigen::Vector3d> lidar_points;
            lidar_points.reserve(cloud->points.size());
            for (const auto& pt : cloud->points) {
                lidar_points.emplace_back(pt.x, pt.y, pt.z);
            }
            
            // 执行深度补全
            geo_output = geometry_head->process(image_rgb, lidar_points, R_cw, t_cw);
            
            std::cout << "\033[1;35m Depth completed,\033[0m";
            // ========== 🔧 新增：存储深度先验到 Dataset ==========
        // 这样后续 optimize() 中所有训练视图都能使用几何约束
        if (geo_output.dense_depth.numel() > 0 && geo_output.uncertainty_map.numel() > 0) {
            int keyframe_idx = static_cast<int>(dataset->train_cameras_.size());  // 注意：此时还没有 addFrame 完成
            // 如果 train_cameras_ 已经在 addFrame 中添加了，则使用 size() - 1
            // 根据代码流程，addFrame 在前面已调用，所以用 size() - 1
            keyframe_idx = static_cast<int>(dataset->train_cameras_.size()) - 1;
            dataset->storePriorDepth(keyframe_idx, geo_output.dense_depth, geo_output.uncertainty_map,geo_output.sparse_depth);
    }
    // ========== 🔧 新增结束 ==========
        }
            // // 1. 准备 RGB 图像
            // cv::Mat image = cv_bridge::toCvCopy(cur_frame.image_msg, "bgr8")->image;

            // // 2. 提取位姿
            // Eigen::Quaterniond q(
            //     cur_frame.pose_msg->pose.orientation.w,
            //     cur_frame.pose_msg->pose.orientation.x,
            //     cur_frame.pose_msg->pose.orientation.y,
            //     cur_frame.pose_msg->pose.orientation.z
            // );
            // Eigen::Matrix3d R_cw = q.toRotationMatrix();
            // Eigen::Vector3d t_cw(
            //     cur_frame.pose_msg->pose.position.x,
            //     cur_frame.pose_msg->pose.position.y,
            //     cur_frame.pose_msg->pose.position.z
            // );

            // // 3. 准备点云
            // pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
            // pcl::fromROSMsg(*cur_frame.point_msg, pcl_cloud);
            // std::vector<Eigen::Vector3d> pointcloud;
            // pointcloud.reserve(pcl_cloud.size());
            // for (const auto& pt : pcl_cloud) {
            //     pointcloud.emplace_back(pt.x, pt.y, pt.z);
            // }

            // // 4. ✅ 调用 GeometryHead (内部自动选择 SGSNet 或双边滤波)
            // GeometryOutput geo_output = geo_head->process(image, pointcloud, R_cw, t_cw);

            // // 5. 保存结果 (逻辑保持不变，但变量源变了)
            // std::string save_dir = "/root/catkin_gaussian/src/Gaussian_lic/test";
            // if (!std::filesystem::exists(save_dir)) {
            //     std::filesystem::create_directories(save_dir);
            // }
            // int frame_id = dataset->all_frame_num_ - 1;

            // cv::Mat depth_save;
            // geo_output.dense_depth_cv.convertTo(depth_save, CV_16UC1, 1000.0); // 使用推理出的 dense_depth
            // cv::imwrite(save_dir + "/dense_depth_" + std::to_string(frame_id) + ".png", depth_save);

            // if (!geo_output.uncertainty_cv.empty()) {
            //     cv::Mat uncertainty_save;
            //     uncertainty_save.convertTo(uncertainty_save, CV_8UC1, 255.0);
            //     cv::imwrite(save_dir + "/uncertainty_" + std::to_string(frame_id) + ".png", uncertainty_save);
            // }

        //================================================================version2
            

        if (!gaussians->is_init_)
        {
            /// [3] initialize map
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset, enable_probabilistic ? &geo_output : nullptr);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians, enable_probabilistic ? &geo_output : nullptr);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        }

        /// [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians, enable_probabilistic ? &geo_output : nullptr);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;
    }

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    torch::NoGradGuard no_grad;
    evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    gaussians->saveMap(result_path);

    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
}

int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/points_for_gs", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/pose_for_gs", 10000, poseCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/image_for_gs", 10000, imageCallback);

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);
    std::string result_path;
    nh.param<std::string>("result_path", result_path, "");
    std::string lpips_path;
    nh.param<std::string>("lpips_path", lpips_path, "");

    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        while (!exit_flag) 
        {
            double now = ros::Time::now().toSec();
            if (gaussians_initialized && (now - last_point_time > 1.0)) 
            {
                exit_flag = true;  // exit if no data is received for more than 1 second
            } 
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    ros::spin();

    mapping_process.join();
    monitor_thread.join();
    
    return 0;
}