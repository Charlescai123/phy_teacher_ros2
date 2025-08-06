#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include "phy_teacher.hpp"
#include "phy_teacher_interfaces/msg/tracking_error.hpp"
#include "phy_teacher_interfaces/msg/control_gains.hpp"

using TrackingError = phy_teacher_interfaces::msg::TrackingError;
using ControlGains = phy_teacher_interfaces::msg::ControlGains;

class ControlGainListener : public rclcpp::Node {
public:
  ControlGainListener()
  : Node("gain_listener")
  {
    subscription_ = this->create_subscription<ControlGains>(
      "/control_gain", 10,
      std::bind(&ControlGainListener::on_gain_update, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ControlGainListener initialized.");
  }

private:
  void on_gain_update(const ControlGains::SharedPtr msg) {

    if (!msg) {
      RCLCPP_ERROR(this->get_logger(), "Received null message in on_gain_update.");
      return;
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        F_kp[i][j]  = msg->kp[i * 6 + j];
        F_kd[i][j]  = msg->kd[i * 6 + j];
      }
    }

    phy_teacher::PHYTeacher::print_matrix6x6(F_kp, "F_kp");
    phy_teacher::PHYTeacher::print_matrix6x6(F_kd, "F_kd");
  }

  phy_teacher::Matrix6x6 F_kp;
  phy_teacher::Matrix6x6 F_kd;
  rclcpp::Subscription<ControlGains>::SharedPtr subscription_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlGainListener>());
  rclcpp::shutdown();
  return 0;
}
