import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


import itertools
import numpy as np
import math

n = 3  # number of locations
k = 1  # number of hiders

S = [i for i in range(n)]  # set of all locations
p = [1/2, 3/4, 2/3]  # probability of succesfull search at each location

# optimal mixed strategy probabilities for searcher and hider (k = 1)
opt_psearch = [0.54585,0.09061,0,0,0,0.36354]
opt_phide = [0.54585,0.18122,0.27293]

# all possible search and hider strategies
search_strats = list(itertools.permutations(S))
hide_strats = list(itertools.combinations(S, k))

# generate the searcher strategy and hiders strategy from the optimal mixed strategies.
search_index = np.random.choice(list(range(math.factorial(n))), p=opt_psearch)
search = list(search_strats[search_index])
H_index = np.random.choice([x[0] for x in itertools.combinations(S, k)], p=opt_phide).item()
H = [H_index]



def generate_launch_description():
    # Configure ROS nodes for launch

    # Setup project paths
    #pkg_project_crazyflie_gazebo = get_package_share_directory('ros_gz_crazyflie_bringup')

    # Setup to launch a crazyflie gazebo simulation from the ros_gz_crazyflie project
    crazyflie_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join('launch', 'launchTest.py'))
    )

    # start a simple mapper node
    mapper = Node(
        package='crazyflie_ros2_multiranger_simple_mapper',
        executable='simple_mapper_multiranger',
        name='simple_mapper',
        output='screen',
        parameters=[
            {'robot_prefix': '/crazyflie'},
            {'use_sim_time': True}
        ]
    )

    # start a wall following node with a delay of 5 seconds
    mover = Node(
        package='moving_try',
        executable='moving_try',
        name='moving_try',
        output='screen',
        parameters=[
            {'robot_prefix': '/crazyflie'},
            {'use_sim_time': True},
            {'delay': 5.0},
            {'max_turn_rate': 0.7},
            {'max_forward_speed': 0.5},
            {'wall_following_direction': 'right'},
            {'hidder': H[0]},
            {'trials': 10}
        ]
    )

    key = Node(
        package='keyboard',
        executable='keyboard',
        name='key',
        output='screen',
        parameters=[
]
    )

    rviz_config_path = os.path.join(
        'config',
        'sim_mapping2.rviz')

    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            parameters=[{
                "use_sim_time": True
            }]
            )

    return LaunchDescription([
        crazyflie_simulation,
        mapper,
        mover,
        key,
        rviz
        ])
