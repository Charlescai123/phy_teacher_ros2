from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'loop_rate',
            default_value='30',
            description='Frequency (Hz) of the main control loop'
        ),
        DeclareLaunchArgument(
            'core_num',
            default_value='1',
            description='CPU core to pin the node using taskset'
        ),
        DeclareLaunchArgument(
            'print_process_time',
            default_value='false',
            description='Whether to print the process time'
        ),
        Node(
            package='phy_teacher_core',
            executable='phy_teacher_node',
            name='phy_teacher_node',
            output='screen',
            prefix=PythonExpression([
                "'taskset -c ' + str(",
                LaunchConfiguration('core_num'),
                ")"
            ]),
            parameters=[
                {'loop_rate': LaunchConfiguration('loop_rate')},
                {'print_process_time': LaunchConfiguration('print_process_time')}
            ],
        )
    ])
