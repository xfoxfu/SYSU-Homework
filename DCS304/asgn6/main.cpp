// clang-format off
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;
constexpr int MAX_POINTS = 10;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < MAX_POINTS) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f de_Casteljau(const std::vector<cv::Point2f> &control_points, float t) 
{
    // clang-format on
    if (control_points.size() == 1)
        return control_points[0];

    std::vector<cv::Point2f> new_points;
    for (int i = 0; i < control_points.size() - 1; ++i)
    {
        new_points.push_back(t * control_points[i] + (1 - t) * control_points[i + 1]);
    }
    return de_Casteljau(new_points, t);
    // clang-format off
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // clang-format on
    // recursive Bezier algorithm.
    constexpr float DELTA = 0.001;
    for (float t = 0; t <= 1; t += DELTA)
    {
        auto point = de_Casteljau(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
    // clang-format off
}

int main() 
{
    cv::Mat window = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for(size_t i = 0;i < control_points.size();++i)
        {
            auto &point = control_points[i];
            cv::circle(window, point, 3, {255, 255, 255}, 3);
            if(i + 1 < control_points.size())
            {
                auto &next = control_points[i+1];
                cv::line(window, point, next, {255, 255, 255}, 1);
            }
        }

        if (control_points.size() == MAX_POINTS) 
        {
            // naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
// clang-format on
