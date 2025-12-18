# ROS Docker shortcut for communication between Raspberry Pis and Hailo Hat integrated in Docker
start_ros_hailo() {
    # CONFIG: The IP of the Master (Pi 4) 
    MASTER_IP="172.20.10.4"
    
    # Grab the actual Wi-Fi IP of this Pi 5 to avoid Docker bridge IP issues 
    MY_IP=$(hostname -I | awk '{print $1}')
    MY_HOSTNAME=$(hostname)

    echo "Starting Hailo Docker (SUDO Mode)..."
    echo "MASTER IP: $MASTER_IP"
    echo "CLIENT IP: $MY_IP ($MY_HOSTNAME)"

    # THE DOCKER COMMAND
    sudo docker run -it --rm --net=host --ipc=host --privileged --device /dev/hailo0:/dev/hailo0 -v ~/hailo-rpi5-examples:/hailo_examples --add-host $MY_HOSTNAME:127.0.0.1 --add-host jostoelz:$MASTER_IP -e ROS_MASTER_URI=http://$MASTER_IP:11311 -e ROS_IP=$MY_IP ros-hailo-noetic:v1
}

