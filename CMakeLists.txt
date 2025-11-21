cmake_minimum_required(VERSION 3.0.2)
project(duckiebot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  duckietown_msgs
)

# catkin_python_setup()

catkin_package()
