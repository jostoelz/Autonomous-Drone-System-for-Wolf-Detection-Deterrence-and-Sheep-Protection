# ROS Docker shortcut for communication between Raspberry Pis and Hailo Hat integrated in Docker 
alias start_ros_hailo='docker run -it --rm \
    --net=host \
    --privileged \
    --device /dev/hailo0:/dev/hailo0 \
    -v ~/hailo-rpi5-examples:/hailo_examples \
    --add-host jostoelz:172.20.10.4 \
    -e ROS_MASTER_URI=http://172.20.10.4:11311 \
    -e ROS_IP=172.20.10.7 \
    ros-hailo-noetic:v1'

