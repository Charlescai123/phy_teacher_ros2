// controller_node.cpp

#include "rclcpp/rclcpp.hpp"
#include "phy_teacher_interfaces/msg/tracking_error.hpp"
#include "phy_teacher_interfaces/msg/control_gains.hpp"
#include "phy_teacher.hpp"

class PhyTeacherNode : public rclcpp::Node {
public:
  PhyTeacherNode() : Node("phy_teacher_node") {
    // Declare parameter and get loop rate
    this->declare_parameter<int>("loop_rate", 30);
    this->declare_parameter<bool>("print_process_time", false);
    
    int loop_rate_hz = this->get_parameter("loop_rate").as_int();
    this->get_parameter("print_process_time", print_process_time_);

    RCLCPP_INFO(this->get_logger(), "Loop rate: %d Hz", loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "Print process time: %s", print_process_time_ ? "true" : "false");

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(1000 / loop_rate_hz),
      std::bind(&PhyTeacherNode::timer_callback, this)
    );

    sub_ = this->create_subscription<phy_teacher_interfaces::msg::TrackingError>(
      "/tracking_err", loop_rate_hz,
      std::bind(&PhyTeacherNode::data_callback, this, std::placeholders::_1)
    );

    pub_ = this->create_publisher<phy_teacher_interfaces::msg::ControlGains>(
      "/control_gain", loop_rate_hz
    );

    RCLCPP_INFO(this->get_logger(), "Phy-Teacher Safety Controller Node initialized.");
  }

private:
  void data_callback(const phy_teacher_interfaces::msg::TrackingError::SharedPtr msg) {
    for (int i = 0; i < 10 && i < msg->tracking_err.size(); ++i) {
      safety_controller_.tracking_err[i] = msg->tracking_err[i];
    }
    got_msg_ = true;
  }

  void timer_callback() {
  if (!got_msg_) return;

  rclcpp::Time start_time;
  if (print_process_time_)
    start_time = this->now();

  // Update controller
  safety_controller_.update();

  // Generate message
  phy_teacher_interfaces::msg::ControlGains gain_msg;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      gain_msg.kp[i * 6 + j] = safety_controller_.F_kp[i][j];
      gain_msg.kd[i * 6 + j] = safety_controller_.F_kd[i][j];
    }
  }

  // Publish message
  pub_->publish(gain_msg);

  if (print_process_time_) {
    rclcpp::Time end_time = this->now();
    double duration_ms = (end_time - start_time).seconds() * 1000.0;
    RCLCPP_INFO(this->get_logger(), "Published control gains. Process time: %.3f ms", duration_ms);
  }
}

  bool got_msg_ = false;
  bool print_process_time_ = false;
  phy_teacher::PHYTeacher safety_controller_;

  rclcpp::Subscription<phy_teacher_interfaces::msg::TrackingError>::SharedPtr sub_;
  rclcpp::Publisher<phy_teacher_interfaces::msg::ControlGains>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PhyTeacherNode>());
  rclcpp::shutdown();
  return 0;
}
