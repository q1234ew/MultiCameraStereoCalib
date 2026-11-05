#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace multicamera {

struct Camera {
    cv::Mat K, D;
    std::string model;
    int width = 0, height = 0;

    void load(const cv::FileNode& node) {
        node["K"] >> K;
        node["D"] >> D;
        node["model"] >> model;
        node["width"] >> width;
        node["height"] >> height;
        if (K.size() != cv::Size(3, 3) || K.type() != CV_64F ||
            D.type() != CV_64F || width <= 0 || height <= 0 ||
            !cv::checkRange(K) || !cv::checkRange(D) ||
            K.at<double>(0, 0) <= 0 || K.at<double>(1, 1) <= 0 ||
            (D.rows != 1 && D.cols != 1))
            throw std::runtime_error("Invalid camera intrinsics");
        const auto n = D.total();
        if (!((model == "fisheye" && n == 4) ||
              (model == "pinhole" && (n == 4 || n == 5 || n == 8 || n == 12 || n == 14))))
            throw std::runtime_error("Unsupported model/distortion count");
    }

    cv::Point2d project(const cv::Vec3d& point) const {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
            !std::isfinite(point[2]) || point[2] <= 0)
            throw std::runtime_error("Projection requires a finite point in front of camera");
        std::vector<cv::Point3d> xyz = {{point[0], point[1], point[2]}};
        std::vector<cv::Point2d> pixels;
        if (model == "fisheye")
            cv::fisheye::projectPoints(xyz, pixels, cv::Vec3d(), cv::Vec3d(), K, D);
        else
            cv::projectPoints(xyz, cv::Vec3d(), cv::Vec3d(), K, D, pixels);
        return pixels.at(0);
    }

    cv::Vec3d unproject(const cv::Point2d& pixel, double optical_depth_mm) const {
        if (!std::isfinite(optical_depth_mm) || optical_depth_mm <= 0 ||
            !std::isfinite(pixel.x) || !std::isfinite(pixel.y))
            throw std::runtime_error("Positive optical-axis depth in mm is required");
        std::vector<cv::Point2d> input = {pixel}, rays;
        if (model == "fisheye")
            cv::fisheye::undistortPoints(input, rays, K, D);
        else
            cv::undistortPoints(input, rays, K, D, cv::noArray(), cv::noArray(),
                                cv::TermCriteria(3, 100, 1e-12));
        return cv::Vec3d(rays[0].x, rays[0].y, 1.) * optical_depth_mm;
    }
};

class Calibration {
public:
    Camera left, right, ir;
    cv::Mat R_ir_left, t_ir_left, R_left_right, t_left_right, R1, R2, P1, P2, Q;
    std::string device_id;

    explicit Calibration(const std::string& path) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("Cannot open calibration");
        std::string unit, version, stereo_direction, ir_direction;
        fs["version"] >> version;
        fs["translation_unit"] >> unit;
        fs["device_id"] >> device_id;
        fs["stereo_transform_direction"] >> stereo_direction;
        fs["ir_transform_direction"] >> ir_direction;
        if (version != "multicamera_rv1126b_v1" || unit != "mm" || device_id.empty() ||
            stereo_direction != "left_to_right" || ir_direction != "ir_to_left")
            throw std::runtime_error("Unsupported schema, unit, device or transform direction");
        left.load(fs["camera_intrinsics"]["left"]);
        right.load(fs["camera_intrinsics"]["right"]);
        ir.load(fs["camera_intrinsics"]["ir"]);
        if (left.width != right.width || left.height != right.height || left.model != right.model)
            throw std::runtime_error("Stereo camera size/model mismatch");
        readTransform(fs["extrinsics"]["T_IR_L"], R_ir_left, t_ir_left);
        readTransform(fs["extrinsics"]["T_RL"], R_left_right, t_left_right);
        for (auto item : std::vector<std::pair<std::string, cv::Mat*>>{
                 {"R1", &R1}, {"R2", &R2}, {"P1", &P1}, {"P2", &P2}, {"Q", &Q}})
            fs["stereo_rectify"][item.first] >> *item.second;
        if (R1.size() != cv::Size(3, 3) || R2.size() != cv::Size(3, 3) ||
            R1.type() != CV_64F || R2.type() != CV_64F ||
            P1.type() != CV_64F || P2.type() != CV_64F || Q.type() != CV_64F ||
            cv::norm(R1.t() * R1 - cv::Mat::eye(3, 3, CV_64F)) > 1e-6 ||
            cv::norm(R2.t() * R2 - cv::Mat::eye(3, 3, CV_64F)) > 1e-6 ||
            std::abs(cv::determinant(R1) - 1.) > 1e-6 ||
            std::abs(cv::determinant(R2) - 1.) > 1e-6 || P1.size() != cv::Size(4, 3) ||
            P2.size() != cv::Size(4, 3) || Q.size() != cv::Size(4, 4) ||
            !cv::checkRange(R1) || !cv::checkRange(R2) || !cv::checkRange(P1) || !cv::checkRange(P2) ||
            !cv::checkRange(Q) || cv::norm(t_left_right) <= 0)
            throw std::runtime_error("Invalid rectification/baseline");
    }

    cv::Point2d irToRgb(cv::Point2d pixel, double ir_depth_mm) const {
        return left.project(transform(R_ir_left, t_ir_left, ir.unproject(pixel, ir_depth_mm)));
    }

    cv::Point2d rgbToIr(cv::Point2d pixel, double rgb_depth_mm) const {
        const auto xyz = left.unproject(pixel, rgb_depth_mm);
        return ir.project(transform(R_ir_left.t(), -R_ir_left.t() * t_ir_left, xyz));
    }

private:
    static void readTransform(const cv::FileNode& n, cv::Mat& R, cv::Mat& t) {
        n["R"] >> R;
        n["t"] >> t;
        if (R.size() != cv::Size(3, 3) || R.type() != CV_64F || t.total() != 3 ||
            t.type() != CV_64F || !cv::checkRange(R) || !cv::checkRange(t) ||
            cv::norm(R.t() * R - cv::Mat::eye(3, 3, CV_64F)) > 1e-6 ||
            std::abs(cv::determinant(R) - 1.) > 1e-6)
            throw std::runtime_error("Invalid rigid transform");
        t = t.reshape(1, 3);
    }

    static cv::Vec3d transform(const cv::Mat& R, const cv::Mat& t, const cv::Vec3d& xyz) {
        cv::Mat out = R * cv::Mat(xyz) + t;
        return cv::Vec3d(out.at<double>(0), out.at<double>(1), out.at<double>(2));
    }
};

}  // namespace multicamera
