# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node



def generate_launch_description():
    # Configure ROS nodes for launch

    # Setup project paths
    #pkg_project_bringup = get_package_share_directory('ros_gz_crazyflie_bringup')
    #pkg_project_gazebo = get_package_share_directory('ros_gz_crazyflie_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_model_path = os.getenv('GZ_SIM_RESOURCE_PATH')

    # Load the SDF file from "description" package
    sdf_file  =  os.path.join(gz_model_path, 'crazyflieH', 'model.sdf')
    with open(sdf_file, 'r') as infp:
        robot_desc = infp.read()

    

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': PathJoinSubstitution([
            'depot.sdf -r'
        ])}.items(),
    )


    gz_drone = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_spawn_model.launch.py')),
        launch_arguments={'world': 'demo','model_string': robot_desc.replace("$name","crazyflie"), 'entity_name': 'crazyflie', 'x': '0.0', 'z': '0.5', 'topic':'crazyflie1'}.items(),
    )

    gz_drone1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_spawn_model.launch.py')),
        launch_arguments={'world': 'demo','model_string': robot_desc.replace("$name","crazyflie1"), 'entity_name': 'crazyflie1', 'x': '0.5', 'z': '0.5', 'topic':'crazyflie1'}.items(),
    )


    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join('config', 'topic_bridge.yaml'),
        }],

        output='screen'
    )

    image = Node(
    package='ros_gz_image',
    executable='image_bridge',
    arguments=['crazyflie/camera/image_raw'],
    output='screen',
)



    cv = Node(
            package='search_det',
            executable='search_det',
            output='screen',
            parameters=[{'robot_prefix': '/crazyflie'}]
            )


    image1 = Node(
    package='ros_gz_image',
    executable='image_bridge',
    arguments=['crazyflie1/camera/image_raw'],
    output='screen',
)

    control = Node(
        package='ros_gz_crazyflie_control',
        executable='control_services',
        name='control',
        output='screen',
        parameters=[
            {'hover_height': 1.5},
            {'robot_prefix': '/crazyflie'},
            {'incoming_twist_topic': '/cmd_vel'},
            {'max_ang_z_rate': 0.4},
        ]
    )

    control1 = Node(
        package='ros_gz_crazyflie_control',
        executable='control_services',
        name='control1',
        output='screen',
        parameters=[
            {'hover_height': 1.5},
            {'robot_prefix': '/crazyflie1'},
            {'incoming_twist_topic': '/cmd_vel'},
            {'max_ang_z_rate': 0.4},
        ]
    )

    return LaunchDescription([
        gz_sim,
        gz_drone,
        #gz_drone1,
        bridge,
        image,
        #image1,
        control,
        #control1
        ])