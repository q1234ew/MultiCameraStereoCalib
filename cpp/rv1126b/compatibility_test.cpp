#include "calibration_adapter.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: compatibility_test calibration.yml vectors.yml");
        multicamera::Calibration calib(argv[1]);
        cv::FileStorage fs(argv[2], cv::FileStorage::READ);
        cv::Mat ir_xyz, rgb_xyz, ir_pixels, rgb_pixels;
        fs["ir_xyz_mm"] >> ir_xyz;
        fs["rgb_xyz_mm"] >> rgb_xyz;
        fs["ir_pixels"] >> ir_pixels;
        fs["rgb_pixels"] >> rgb_pixels;
        if (ir_xyz.rows < 27 || ir_xyz.cols != 3 || rgb_xyz.size() != ir_xyz.size() ||
            ir_pixels.rows != ir_xyz.rows || ir_pixels.cols != 2 ||
            rgb_pixels.size() != ir_pixels.size()) throw std::runtime_error("Invalid test vectors");
        double error = 0.;
        for (int i = 0; i < ir_xyz.rows; ++i) {
            cv::Vec3d ir(ir_xyz.at<double>(i, 0), ir_xyz.at<double>(i, 1), ir_xyz.at<double>(i, 2));
            cv::Vec3d rgb(rgb_xyz.at<double>(i, 0), rgb_xyz.at<double>(i, 1), rgb_xyz.at<double>(i, 2));
            cv::Point2d pi(ir_pixels.at<double>(i, 0), ir_pixels.at<double>(i, 1));
            cv::Point2d pr(rgb_pixels.at<double>(i, 0), rgb_pixels.at<double>(i, 1));
            error = std::max({error, cv::norm(calib.ir.project(ir) - pi),
                             cv::norm(calib.left.project(rgb) - pr),
                             cv::norm(calib.irToRgb(pi, ir[2]) - pr),
                             cv::norm(calib.rgbToIr(pr, rgb[2]) - pi)});
        }
        // Q must reconstruct millimetres, not the source file's metre-scale depth.
        cv::Mat X = (cv::Mat_<double>(4, 1) << 10., 20., 1000., 1.);
        cv::Mat l = calib.P1 * X, r = calib.P2 * X;
        const int axis = std::abs(calib.P2.at<double>(0, 3)) >
                         std::abs(calib.P2.at<double>(1, 3)) ? 0 : 1;
        double u = l.at<double>(0) / l.at<double>(2);
        double v = l.at<double>(1) / l.at<double>(2);
        double disparity = l.at<double>(axis) / l.at<double>(2) -
                           r.at<double>(axis) / r.at<double>(2);
        cv::Mat q_input = (cv::Mat_<double>(4, 1) << u, v, disparity, 1.);
        cv::Mat back = calib.Q * q_input;
        double depth_error = std::abs(back.at<double>(2) / back.at<double>(3) - 1000.);
        if (!std::isfinite(error) || error > 1e-5 || !std::isfinite(depth_error) || depth_error > 1e-5)
            throw std::runtime_error("Projection or millimetre-depth consistency failed");
        std::cout << std::setprecision(12) << "{\"status\":\"passed\",\"opencv\":\""
                  << CV_VERSION << "\",\"vectors\":" << ir_xyz.rows
                  << ",\"max_error_px\":" << error << ",\"depth_error_mm\":" << depth_error
                  << ",\"physical_validation\":false,\"activation\":false}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
