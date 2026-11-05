#include "calibration_adapter.hpp"
#include "vendor/httplib.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ifaddrs.h>
#ifdef __linux__
#include <netpacket/packet.h>
#else
#include <net/if_dl.h>
#endif
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

using Clock = std::chrono::steady_clock;

static std::string json(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    return out.str() + '"';
}

static bool localMacMatches(const std::string& expected) {
    ifaddrs* addresses = nullptr;
    if (getifaddrs(&addresses)) return false;
    bool found = false;
    for (auto* p = addresses; p; p = p->ifa_next) {
        if (!p->ifa_addr) continue;
        const unsigned char* bytes = nullptr;
        int count = 0;
#ifdef __linux__
        if (p->ifa_addr->sa_family == AF_PACKET) {
            auto* link = reinterpret_cast<sockaddr_ll*>(p->ifa_addr);
            bytes = link->sll_addr;
            count = link->sll_halen;
        }
#else
        if (p->ifa_addr->sa_family == AF_LINK) {
            auto* link = reinterpret_cast<sockaddr_dl*>(p->ifa_addr);
            bytes = reinterpret_cast<const unsigned char*>(LLADDR(link));
            count = link->sdl_alen;
        }
#endif
        if (count != 6) continue;
        std::ostringstream mac;
        for (int i = 0; i < count; ++i)
            mac << (i ? ":" : "") << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
        if (mac.str() == expected) found = true;
    }
    freeifaddrs(addresses);
    return found;
}

struct Config {
    std::string calibration, device, mac, bind;
    int port = 8090, left = 8080, right = 8082, ir = 8081;
    explicit Config(const char* path) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) throw std::runtime_error("Cannot open service config");
        fs["calibration"] >> calibration;
        fs["device_id"] >> device;
        fs["device_mac"] >> mac;
        fs["bind"] >> bind;
        fs["port"] >> port;
        fs["left_port"] >> left;
        fs["right_port"] >> right;
        fs["ir_port"] >> ir;
        if (calibration.empty() || device.empty() || mac.size() != 17 ||
            (bind != "127.0.0.1" && bind != "0.0.0.0") || port < 1024 || port > 65535 ||
            left < 1024 || left > 65535 || right < 1024 || right > 65535 ||
            ir < 1024 || ir > 65535 || left == right || left == ir || right == ir ||
            port == left || port == right || port == ir)
            throw std::runtime_error("Invalid config/bind/ports");
        if (!localMacMatches(mac)) throw std::runtime_error("Device MAC binding mismatch");
    }
};

static void clientTimeouts(httplib::Client& client) {
    client.set_connection_timeout(0, 400000);
    client.set_read_timeout(1, 0);
    client.set_write_timeout(1, 0);
    client.set_follow_location(false);
    client.set_keep_alive(false);
}

// Validate the JPEG dimensions before OpenCV allocates the decoded image.
static bool jpegSize(const std::string& bytes, int w, int h) {
    if (bytes.size() < 4 || uint8_t(bytes[0]) != 255 || uint8_t(bytes[1]) != 216) return false;
    size_t i = 2;
    while (i + 4 <= bytes.size()) {
        if (uint8_t(bytes[i++]) != 255) return false;
        while (i < bytes.size() && uint8_t(bytes[i]) == 255) ++i;
        if (i + 3 > bytes.size()) return false;
        unsigned marker = uint8_t(bytes[i++]);
        if (marker == 218 || marker == 217) return false;
        size_t n = uint8_t(bytes[i]) * 256 + uint8_t(bytes[i + 1]);
        if (n < 2 || n > bytes.size() - i) return false;
        if ((marker >= 192 && marker <= 195) || (marker >= 197 && marker <= 199) ||
            (marker >= 201 && marker <= 203) || (marker >= 205 && marker <= 207)) {
            return n >= 8 && uint8_t(bytes[i + 3]) * 256 + uint8_t(bytes[i + 4]) == h &&
                   uint8_t(bytes[i + 5]) * 256 + uint8_t(bytes[i + 6]) == w;
        }
        i += n;
    }
    return false;
}

static cv::Mat snapshot(int port, const multicamera::Camera& camera) {
    httplib::Client client("127.0.0.1", port);
    clientTimeouts(client);
    std::string bytes;
    const auto start = Clock::now();
    auto response = client.Get("/snapshot", [&](const char* data, size_t n) {
        if (n > 4 * 1024 * 1024 - bytes.size() || Clock::now() - start > std::chrono::seconds(2))
            return false;
        bytes.append(data, n);
        return true;
    });
    if (!response || response->status != 200)
        throw std::runtime_error("Raw snapshot unavailable on port " + std::to_string(port));
    if (!jpegSize(bytes, camera.width, camera.height))
        throw std::runtime_error("Raw JPEG dimensions/format do not match calibration");
    std::vector<uchar> input(bytes.begin(), bytes.end());
    auto frame = cv::imdecode(input, cv::IMREAD_COLOR);
    if (frame.empty() || frame.cols != camera.width || frame.rows != camera.height)
        throw std::runtime_error("Raw JPEG decode failed");
    return frame;
}

static double number(const httplib::Request& req, const char* key, double low, double high) {
    if (req.get_param_value_count(key) != 1) throw std::invalid_argument(std::string("Required once: ") + key);
    const auto value = req.get_param_value(key);
    size_t end = 0;
    double result;
    try { result = std::stod(value, &end); }
    catch (...) { throw std::invalid_argument(std::string("Invalid number: ") + key); }
    if (end != value.size() || !std::isfinite(result) || result < low || result > high)
        throw std::invalid_argument(std::string("Out of range: ") + key);
    return result;
}

static void sendJpeg(httplib::Response& res, const cv::Mat& frame) {
    std::vector<uchar> encoded;
    if (!cv::imencode(".jpg", frame, encoded, {cv::IMWRITE_JPEG_QUALITY, 90}))
        throw std::runtime_error("JPEG encoding failed");
    res.set_content(reinterpret_cast<const char*>(encoded.data()), encoded.size(), "image/jpeg");
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: calibration_service CONFIG.yml");
        cv::setNumThreads(1);
        Config cfg(argv[1]);
        multicamera::Calibration calib(cfg.calibration);
        if (calib.device_id != cfg.device) throw std::runtime_error("Calibration device ID mismatch");
        std::ifstream input(cfg.calibration, std::ios::binary);
        std::string yaml((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        cv::Mat lm1, lm2, rm1, rm2;
        const cv::Size size(calib.left.width, calib.left.height);
        auto maps = [&](const multicamera::Camera& camera, const cv::Mat& R, const cv::Mat& P,
                        cv::Mat& m1, cv::Mat& m2) {
            if (camera.model == "fisheye")
                cv::fisheye::initUndistortRectifyMap(camera.K, camera.D, R, P, size, CV_16SC2, m1, m2);
            else cv::initUndistortRectifyMap(camera.K, camera.D, R, P, size, CV_16SC2, m1, m2);
        };
        maps(calib.left, calib.R1, calib.P1, lm1, lm2);
        maps(calib.right, calib.R2, calib.P2, rm1, rm2);
        std::mutex compute;
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(2, 4); };
        server.set_read_timeout(2).set_write_timeout(2).set_payload_max_length(1024);
        server.set_keep_alive_max_count(2);
        server.set_pre_routing_handler([](const auto&, auto& res) {
            res.set_header("Cache-Control", "no-store");
            res.set_header("X-Calibration-Qualification", "candidate-not-physically-validated");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        auto guard = [&](auto handler) {
            return [&, handler](const httplib::Request& req, httplib::Response& res) {
                std::unique_lock<std::mutex> lock(compute, std::try_to_lock);
                if (!lock.owns_lock()) {
                    res.status = 503;
                    res.set_content("{\"error\":\"processor_busy\"}", "application/json");
                    return;
                }
                try { handler(req, res); }
                catch (const std::invalid_argument& error) {
                    res.status = 400;
                    res.set_content("{\"error\":" + json(error.what()) + "}", "application/json");
                }
                catch (const std::exception& error) {
                    std::cerr << error.what() << std::endl;
                    res.status = 503;
                    res.set_content("{\"error\":" + json(error.what()) + "}", "application/json");
                }
            };
        };
        server.Get("/health", [&](const auto&, auto& res) {
            res.set_content("{\"service\":\"multicamera-calibration\",\"parameters_loaded\":true,"
                            "\"device_id\":" + json(cfg.device) +
                            ",\"physical_validation\":false,\"raw_streams_modified\":false,"
                            "\"fusion_mode\":\"explicit_depth_plane_only\","
                            "\"source_status_endpoint\":\"/api/sources\"}", "application/json");
        });
        server.Get("/api/calibration", [&](const auto&, auto& res) {
            res.set_content(yaml, "application/x-yaml");
        });
        server.Get("/api/raw-state", [&](const auto& req, auto& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            try {
                const double requested = number(req, "port", 1024, 65535);
                const int port = int(requested);
                if (requested != port) throw std::invalid_argument("Port must be an integer");
                if (port != cfg.left && port != cfg.right && port != cfg.ir)
                    throw std::invalid_argument("Only configured raw ports are allowed");
                httplib::Client client("127.0.0.1", port);
                clientTimeouts(client);
                std::string body;
                const auto started = Clock::now();
                const auto reply = client.Get("/state", [&](const char* data, size_t count) {
                    if (count > 65536 - body.size() || Clock::now() - started > std::chrono::seconds(2))
                        return false;
                    body.append(data, count);
                    return true;
                });
                if (!reply || reply->status != 200 || body.empty()) {
                    res.status = 503;
                    res.set_content("{\"error\":\"raw_state_unavailable\"}", "application/json");
                } else res.set_content(body, "application/json");
            } catch (const std::invalid_argument& error) {
                res.status = 400;
                res.set_content("{\"error\":" + json(error.what()) + "}", "application/json");
            }
        });
        auto page = [&](const char* name) {
            return [&, name](const auto&, auto& res) {
                const auto path = std::filesystem::path(argv[1]).parent_path() / name;
                std::ifstream file(path, std::ios::binary);
                if (!file) { res.status = 404; return; }
                res.set_content(std::string(std::istreambuf_iterator<char>(file), {}), "text/html; charset=utf-8");
            };
        };
        server.Get("/", page("panel.html"));
        server.Get("/panel", page("panel.html"));
        server.Get("/dashboard", page("dashboard.html"));
        server.Get("/api/sources", guard([&](const auto&, auto& res) {
            std::ostringstream out;
            out << "{\"hardware_synchronized\":false,\"sources\":[";
            int ports[] = {cfg.left, cfg.right, cfg.ir};
            const multicamera::Camera* cameras[] = {&calib.left, &calib.right, &calib.ir};
            const char* names[] = {"left", "right", "ir"};
            for (int i = 0; i < 3; ++i) {
                bool available = true;
                std::string reason;
                try { snapshot(ports[i], *cameras[i]); }
                catch (const std::exception& e) { available = false; reason = e.what(); }
                if (i) out << ',';
                out << "{\"camera\":" << json(names[i]) << ",\"port\":" << ports[i]
                    << ",\"snapshot_valid\":" << (available ? "true" : "false")
                    << ",\"reason\":" << json(reason) << '}';
            }
            out << "]}";
            res.set_content(out.str(), "application/json");
        }));
        server.Get("/api/project", guard([&](const auto& req, auto& res) {
            auto from = req.get_param_value("from");
            if (req.get_param_value_count("from") != 1 || (from != "rgb" && from != "ir"))
                throw std::invalid_argument("from must be rgb (raw left) or ir");
            const auto& source = from == "rgb" ? calib.left : calib.ir;
            const auto& target = from == "rgb" ? calib.ir : calib.left;
            cv::Point2d pixel(number(req, "u", 0, source.width - 1), number(req, "v", 0, source.height - 1));
            double depth = number(req, "depth_mm", 50, 100000);
            const auto point = from == "rgb" ? calib.rgbToIr(pixel, depth) : calib.irToRgb(pixel, depth);
            if (!std::isfinite(point.x) || !std::isfinite(point.y))
                throw std::invalid_argument("Projection is nonfinite");
            std::ostringstream out;
            out << std::setprecision(12) << "{\"u\":" << point.x << ",\"v\":" << point.y
                << ",\"depth_unit\":\"mm\",\"inside_target\":"
                << (point.x >= 0 && point.x < target.width && point.y >= 0 && point.y < target.height ? "true" : "false") << '}';
            res.set_content(out.str(), "application/json");
        }));
        server.Get("/api/rectified.jpg", guard([&](const auto& req, auto& res) {
            auto camera = req.get_param_value("camera");
            if (req.get_param_value_count("camera") != 1 || (camera != "left" && camera != "right"))
                throw std::invalid_argument("camera must be left or right");
            bool left = camera == "left";
            auto frame = snapshot(left ? cfg.left : cfg.right, left ? calib.left : calib.right);
            cv::Mat corrected;
            cv::remap(frame, corrected, left ? lm1 : rm1, left ? lm2 : rm2, cv::INTER_LINEAR);
            res.set_header("X-Image-Coordinates", "rectified");
            sendJpeg(res, corrected);
        }));
        server.Get("/api/fusion.jpg", guard([&](const auto& req, auto& res) {
            double depth = number(req, "depth_mm", 50, 100000);
            auto start = Clock::now();
            auto rgb = snapshot(cfg.left, calib.left);
            auto rgbReceived = Clock::now();
            auto ir = snapshot(cfg.ir, calib.ir);
            auto irReceived = Clock::now();
            // A frontoparallel plane at raw RGB optical-axis Z, not a scene depth estimate.
            std::vector<cv::Point2d> pixels, rays;
            pixels.reserve(size.area());
            for (int y = 0; y < size.height; ++y)
                for (int x = 0; x < size.width; ++x) pixels.emplace_back(x, y);
            if (calib.left.model == "fisheye") cv::fisheye::undistortPoints(pixels, rays, calib.left.K, calib.left.D);
            else cv::undistortPoints(pixels, rays, calib.left.K, calib.left.D, cv::noArray(), cv::noArray(), cv::TermCriteria(3, 100, 1e-10));
            cv::Mat rotation = calib.R_ir_left.t();
            cv::Mat translation = -rotation * calib.t_ir_left;
            std::vector<cv::Point3d> points;
            std::vector<uchar> valid;
            points.reserve(rays.size());
            for (const auto& ray : rays) {
                cv::Vec3d xyz(ray.x * depth, ray.y * depth, depth);
                cv::Mat p = rotation * cv::Mat(xyz) + translation;
                cv::Point3d point(p.at<double>(0), p.at<double>(1), p.at<double>(2));
                bool good = std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) && point.z > 0;
                valid.push_back(good);
                points.push_back(good ? point : cv::Point3d(0, 0, 1));
            }
            std::vector<cv::Point2d> projected;
            if (calib.ir.model == "fisheye") cv::fisheye::projectPoints(points, projected, cv::Vec3d(), cv::Vec3d(), calib.ir.K, calib.ir.D);
            else cv::projectPoints(points, cv::Vec3d(), cv::Vec3d(), calib.ir.K, calib.ir.D, projected);
            cv::Mat mx(size, CV_32F), my(size, CV_32F), mask(size, CV_8U, cv::Scalar(0));
            for (size_t i = 0; i < projected.size(); ++i) {
                const auto& p = projected[i];
                bool good = valid[i] && std::isfinite(p.x) && std::isfinite(p.y) &&
                            p.x >= 0 && p.y >= 0 && p.x < ir.cols - 1 && p.y < ir.rows - 1;
                mx.ptr<float>()[i] = good ? float(p.x) : -1;
                my.ptr<float>()[i] = good ? float(p.y) : -1;
                mask.ptr<uchar>()[i] = good ? 255 : 0;
            }
            cv::Mat warped, mixed;
            cv::remap(ir, warped, mx, my, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            cv::addWeighted(rgb, 0.5, warped, 0.5, 0, mixed);
            mixed.copyTo(rgb, mask);
            cv::putText(rgb, "PLANE ONLY Z=" + std::to_string(int(depth)) + "mm - NOT SCENE DEPTH", {12, 26}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 255}, 2);
            res.set_header("X-Fusion-Mode", "raw-rgb-constant-optical-depth-plane");
            res.set_header("X-Depth-Mm", std::to_string(depth));
            res.set_header("X-Source-Receive-Gap-Ms", std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(irReceived - rgbReceived).count()));
            res.set_header("X-Hardware-Synchronized", "false");
            res.set_header("X-Elapsed-Ms", std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count()));
            sendJpeg(res, rgb);
        }));
        server.set_logger([](const auto& req, const auto& res) {
            std::cerr << req.method << ' ' << req.path << " status=" << res.status << std::endl;
        });
        std::cerr << "parameters_loaded device=" << cfg.device << " port=" << cfg.port
                  << " physical_validation=false raw_streams_modified=false" << std::endl;
        if (!server.listen(cfg.bind, cfg.port)) throw std::runtime_error("Listen failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
