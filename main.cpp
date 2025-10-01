#include <ceres/ceres.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int detectionball(cv::Mat img, std::vector<cv::Point2f> &centers, cv::Scalar lowerb, cv::Scalar upperb);
void exception_situation(int code);
std::array<double, 4> fitTrajectoryWithCeres(const std::vector<cv::Point2f> &centers, int frequency, double params[4], double x0, double y0);
std::vector<float> init_v0_calculate(const std::vector<cv::Point2f> &centers, int frequency);

struct TrajectoryResidual
{
    TrajectoryResidual(double t, double x_obs, double y_obs,
                       double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}

    template <typename T>
    bool operator()(const T *const params, T *residuals) const
    {
        const T vx0 = T(params[0]);
        const T vy0 = T(params[1]);
        const T k = T(params[2]);
        const T g = T(params[3]);

        T dt = T(t_);
        T exp_term = exp(-k * dt);

        T x_pred = T(x0_) + vx0/ k * (T(1.0) - exp_term);
        T y_pred = T(y0_) + (vy0 - g / k) / k * (T(1.0) - exp_term) + g / k * dt;

        residuals[0] = T(x_obs_) - x_pred;
        residuals[1] = T(y_obs_) - y_pred;
        return true;
    }

private:
    double t_, x_obs_, y_obs_, x0_, y0_;
};

int main(int, char **)
{
    cv::VideoCapture cap("../../TASK03/video.mp4"); // 打开视频
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }
    cv::Mat frame;
    std::vector<cv::Point2f> centers;
    int fps = cap.get(cv::CAP_PROP_FPS);
    while (true)
    {
        cap >> frame; // 读取下一帧
        if (frame.empty() || cv::waitKey(30) >= 0)
        { // 等待30ms，按任意键退出
            break;
        }

        cv::Scalar lowerb = cv::Scalar(90, 62, 51);
        cv::Scalar upperb = cv::Scalar(102, 255, 255);
        
        int op = detectionball(frame, centers, lowerb, upperb);
        exception_situation(op);
        if (op != 0) break; // 简化处理

    }
    cap.release();
    cv::destroyAllWindows();

    auto v0 = init_v0_calculate(centers, fps);
    std::cout << "Initial velocity estimate: vx0=" << v0[0] << ", vy0=" << v0[1] << std::endl;
    double params[4] = { v0[0],v0[1], 0.1, 100.0}; // 初始参数猜测
    double x0 = centers.front().x;
    double y0 = centers.front().y;
    auto result = fitTrajectoryWithCeres(centers, fps, params,x0, y0);
    std::cout << "parameters: vx0=" <<result[0]
              << ", vy0=" << result[1]
              << ", k=" << result[2]
              << ", g=" << result[3] << std::endl;
    return 0;
}

// 检测蓝色小球，并返回中心坐标
int detectionball(cv::Mat img, std::vector<cv::Point2f> &centers, cv::Scalar lowerb, cv::Scalar upperb)
{
    cv::Mat img_hsv;
    cv::cvtColor(img, img_hsv, cv::COLOR_BGR2HSV); // 转换为HSV颜色空间
    // 颜色过滤，提取蓝色部分
    cv::Mat mask_blue;
    cv::inRange(img_hsv, lowerb, upperb, mask_blue);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_blue, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 处理异常情况
    if (contours.size() == 0)
        return -1;
    else if (contours.size() > 1)
        return -2;
    // 对于单目标拟合问题，省略contours循环处理的部分
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(contours[0], center, radius);
    centers.push_back(center);
    return 0;
}

// 如果出现异常情况，进行相应处理，这里因无异常简化代码
void exception_situation(int code)
{
    switch (code)
    {
    // 由于在本视频里，蓝色球消失后未再出现，因此直接结束循环简化处理
    // 实际中可对消失点进行标记，并等待下一次出现
    case 0:
        break;
    case -1:
        std::cout << "No ball detected in this frame." << std::endl;
        break;
    // 经过正确处理后，contours里只会有一组轮廓，因此认为存在多目标检测即为异常
    // 实际中如有多球同时发射，可对多目标进行分别标记
    case -2:
        std::cout << "Multiple balls detected in this frame." << std::endl;
        break;
    default:
        std::cout << "Unknown error code." << std::endl;
        break;
    }
}

std::array<double, 4> fitTrajectoryWithCeres(
    const std::vector<cv::Point2f> &centers,
    int frequency,
    double params[4],
    double x0,
    double y0)
{
    ceres::Problem problem;
    ceres::LossFunction *loss = new ceres::HuberLoss(1.0);

    int step = 3; // 每隔step个点取一个点进行拟合
    double delta_t = 1.0 / frequency;
    // 加入所有观测点
    for (size_t i = 0; i < centers.size(); i += step)
    {
        double t = i * delta_t;
        double x = centers[i].x;
        double y = centers[i].y;

        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 4>(
                new TrajectoryResidual(t, x, y, x0, y0));

        problem.AddResidualBlock(cost_function, loss, params);
    }

    problem.SetParameterLowerBound(params, 2, 0.01);
    problem.SetParameterUpperBound(params, 2, 1.0);

    problem.SetParameterLowerBound(params, 3, 100.0);
    problem.SetParameterUpperBound(params, 3, 1000.0);

    // Solver 配置
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // 输出拟合参数
    std::cout << summary.BriefReport() << std::endl;
    return {params[0], params[1], params[2], params[3]};
}

//近似以均变速估计初始速度
std::vector<float> init_v0_calculate(const std::vector<cv::Point2f> &centers, int frequency)
{
    std::vector<float> v0;
    float v0_x, v0_y,v1_x,v1_y,v2_x,v2_y;
    float delta_t = 1.0 / frequency;
    v2_x = (centers[2].x - centers[1].x) / delta_t;
    v2_y = (centers[2].y - centers[1].y) / delta_t;
    v1_x = (centers[1].x - centers[0].x) / delta_t;
    v1_y = (centers[1].y - centers[0].y) / delta_t;
    v0_x = 2 * v1_x - v2_x;
    v0_y = 2 * v1_y - v2_y;
    v0.push_back(v0_x);
    v0.push_back(v0_y);
    return v0;
}