import sys
if sys.prefix == '/home/tim/code/ground_school/4_5_ros2/ros_ws/.pixi/envs/default':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tim/code/ground_school/4_5_ros2/ros_ws/install/py_srvcli'
