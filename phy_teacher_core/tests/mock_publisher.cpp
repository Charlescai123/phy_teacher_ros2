#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include "phy_teacher_interfaces/msg/tracking_error.hpp"

using namespace std::chrono_literals;
using TrackingError = phy_teacher_interfaces::msg::TrackingError;

class MockPublisher : public rclcpp::Node {
public:
  MockPublisher()
  : Node("mock_publisher"),
    gen_(rd_()), dist_(-0.1, 0.1)
  {
    publisher_ = this->create_publisher<TrackingError>("/tracking_err", 30);
    timer_ = this->create_wall_timer(100ms, std::bind(&MockPublisher::timer_callback, this));
    RCLCPP_INFO(this->get_logger(), "MockPublisher started.");
  }

private:
void timer_callback() {
  TrackingError msg;
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < 10; ++i) {
    msg.tracking_err[i] = dist_(gen_);
    oss << msg.tracking_err[i];
    if (i < 9) oss << ", ";
  }
  oss << "]";
  publisher_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "Published tracking_error: %s", oss.str().c_str());
}


  rclcpp::Publisher<TrackingError>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dist_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockPublisher>());
  rclcpp::shutdown();
  return 0;
}
