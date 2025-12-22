#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Pose3.hh>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>
#include <memory>
#include <cmath>

namespace gazebo
{
  class AnimatedBox : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/)
    {
        this->model = _model;
        this->world = _model->GetWorld();
        this->model->SetGravityMode(false);

        // ROS initialisieren
        if (!ros::isInitialized())
        {
            int argc = 0;
            char **argv = nullptr;
            ros::init(argc, argv, "animated_box_node");
        }
        this->rosNode = std::make_unique<ros::NodeHandle>("~");
        this->rosPub = this->rosNode->advertise<geometry_msgs::Pose>("animated_box/pose", 10);

        // Gazebo Update Callback
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&AnimatedBox::OnUpdate, this));
    }

    void OnUpdate()
    {
        double t = this->world->SimTime().Double();
        double x = 3.0 * sin(t * M_PI / 5.0);  // 10s Loop
        double y = 2.0 * cos(t * M_PI / 5.0);
        double z = 0.5;

        // Box-Pose in Gazebo setzen
        this->model->SetWorldPose(ignition::math::Pose3d(x, y, z, 0, 0, 0));

        // ROS-Pose veröffentlichen
        geometry_msgs::Pose msg;
        msg.position.x = x;
        msg.position.y = y;
        msg.position.z = z;
        msg.orientation.x = 0;
        msg.orientation.y = 0;
        msg.orientation.z = 0;
        msg.orientation.w = 1;

        this->rosPub.publish(msg);
    }

  private:
    physics::ModelPtr model;
    physics::WorldPtr world;
    event::ConnectionPtr updateConnection;

    // ROS
    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Publisher rosPub;
  };

  GZ_REGISTER_MODEL_PLUGIN(AnimatedBox)
}
