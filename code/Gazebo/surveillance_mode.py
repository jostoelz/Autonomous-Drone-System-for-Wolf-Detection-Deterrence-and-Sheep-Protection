#!/usr/bin/env python3
import rospy
import os 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist, Point, PoseStamped
from mavros_msgs.msg import State
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import DeleteModel 

def spawn_quadrat_marker(route, altitude):
    """
    Spawns red spheres in Gazebo to visualize the planned trajectory.
    Accepts 'route' (list of x,y coordinates) and 'altitude' (z-axis).
    """
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # XML definition for a red sphere
    sdf_sphere = """
    <?xml version='1.0'?>
    <sdf version='1.4'>
      <model name='marker_{}'>
        <static>true</static>
        <link name='link'>
          <visual name='visual'>
            <geometry><sphere><radius>0.3</radius></sphere></geometry> <material>
              <ambient>1 0 0 1</ambient> <diffuse>1 0 0 1</diffuse>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """

    for i, point in enumerate(route):
        # Extracts x and y from the route list, using the altitude variable for z
        # An offset is added to elevate the marker slightly
        pose = Pose(Point(point[0], point[1], altitude + 1), Quaternion(0,0,0,1))
        
        try:
            spawn_client(
                model_name=f"waypoint_{i}",
                model_xml=sdf_sphere.format(i),
                robot_namespace="/",
                initial_pose=pose,
                reference_frame="world"
            )
            rospy.loginfo(f"Marker {i} set at {point} with altitude {altitude}m.")
        except rospy.ServiceException:
            pass

def spawn_colored_ground():
    """
    Deletes the existing ground model (if any) and spawns a new, 
    high-contrast blue surface for better visualization.
    """
    
    # 1. Ensure any previous ground model is removed
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        del_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        del_client("custom_ground")
    except rospy.ServiceException:
        pass 

    # 2. Spawn the new ground model
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    spawn_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # Define a massive ground plane (100x100 meters) colored blue
    sdf_ground = """
    <?xml version='1.0'?>
    <sdf version='1.4'>
      <model name='custom_ground'>
        <static>true</static>
        <link name='link'>
          
          <visual name='visual'>
            <geometry>
              <box>
                <size>100 100 0.1</size> 
              </box>
            </geometry>
            <material>
              <script>
                <uri>file://media/materials/scripts/gazebo.material</uri>
                <name>Gazebo/White</name> 
              </script>
            </material>
          </visual>

          <collision name='collision'> 
            <geometry>
              <box>
                <size>100 100 0.1</size> 
              </box>
            </geometry>
          </collision>

        </link>
      </model>
    </sdf>
    """
    
    # Z-Positioning: The box has a height of 0.1m. The origin is the center.
    # Setting z=0.06 places the bottom surface slightly above the default ground (z=0)
    # to prevent z-fighting (flickering textures).
    pose = Pose(Point(5, 5, 0.06), Quaternion(0,0,0,1)) 

    try:
        spawn_client("custom_ground", sdf_ground, "/", pose, "world")
        rospy.loginfo("Large blue ground plane spawned.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn ground: {e}")
        
class SurveillanceVelocity:
    def __init__(self):
        rospy.init_node("surveillance_node")

        # Publishers
        self.vel_pub = rospy.Publisher("/surveillance_velocity", Twist, queue_size=10)

        # Subscribers
        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        # Route Definition (Square)
        self.route = [(0, 0), (10, 0), (10,10), (0, 10)]
        self.current_target_index = 0

        self.current_position = Point()
        self.current_position.z = 0.0
        self.altitude = 4.0
        self.max_vel = 3.0

        # --- TUNING AREA ---
        
        # X/Y Control parameters
        self.kp_xy = 1.2   
        self.kd_xy = 0.6   
        self.tolerance = 0.3 
        
        # Z-Axis (Altitude) Control parameters
        self.kp_z = 1.5    
        self.ki_z = 0.2    
        self.kd_z = 0.8    
        
        # PID Internal Storage
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dz = 0.0
        self.integral_z = 0.0
        
        self.last_time = rospy.Time.now()
        self.first_run = True
        
        # Data Storage for Plotting
        self.path_x = []
        self.path_y = []
        self.path_z = []
        self.time_data = []
        self.start_time = rospy.Time.now().to_sec()
        
        # Statistics Storage
        self.accuracy_stats = [] 
        self.last_waypoint = (0, 0)

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_position = msg.pose.position
        
        # Store data for plotting
        self.path_x.append(msg.pose.position.x)
        self.path_y.append(msg.pose.position.y)
        self.path_z.append(msg.pose.position.z)
        self.time_data.append(rospy.Time.now().to_sec() - self.start_time)

    def compute_velocity(self):
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        self.last_time = now

        if dt <= 0 or dt > 1.0: dt = 0.033

        # --- 1. Navigation & Target Selection ---
        target = self.route[self.current_target_index]
        dx = target[0] - self.current_position.x
        dy = target[1] - self.current_position.y
        distance = (dx**2 + dy**2)**0.5

        # Waypoint Logic
        if distance < self.tolerance:
            # Statistics Calculation
            prev = self.last_waypoint
            curr = self.route[self.current_target_index]
            segment_length = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
            
            if segment_length > 0:
                rel_accuracy = distance / segment_length
                self.accuracy_stats.append({
                    'index': self.current_target_index,
                    'error_abs': distance,
                    'dist_x': segment_length,
                    'rel_acc': rel_accuracy
                })
                rospy.loginfo(f"STATISTICS: Target {self.current_target_index} reached. Error: {distance:.3f}m")

            self.last_waypoint = curr
            self.current_target_index = (self.current_target_index + 1) % len(self.route)
            
            # Update target immediately for smoothness
            target = self.route[self.current_target_index]
            dx = target[0] - self.current_position.x
            dy = target[1] - self.current_position.y
            self.last_dx = dx
            self.last_dy = dy

        # --- 2. X/Y Control (PD) ---
        if self.first_run:
            d_dx = 0.0
            d_dy = 0.0
        else:
            d_dx = (dx - self.last_dx) / dt
            d_dy = (dy - self.last_dy) / dt
        
        self.last_dx = dx
        self.last_dy = dy

        vel_x = (self.kp_xy * dx) + (self.kd_xy * d_dx)
        vel_y = (self.kp_xy * dy) + (self.kd_xy * d_dy)

        # --- 3. Z-Control (PID) ---
        dz = self.altitude - self.current_position.z
        self.integral_z += dz * dt
        self.integral_z = max(min(self.integral_z, 2.0), -2.0) 
        
        if self.first_run:
            d_dz = 0.0
            self.first_run = False
        else:
            d_dz = (dz - self.last_dz) / dt
        self.last_dz = dz

        vel_z = (self.kp_z * dz) + (self.ki_z * self.integral_z) + (self.kd_z * d_dz)
        
        # Velocity Limits
        vel_x = max(min(vel_x, self.max_vel), -self.max_vel)
        vel_y = max(min(vel_y, self.max_vel), -self.max_vel)
        vel_z = max(min(vel_z, self.max_vel), -self.max_vel)
        
        rospy.loginfo_throttle(0.5, f"Z:{self.current_position.z:.2f} | Cmd:{vel_z:.2f}")

        twist = Twist()
        twist.linear.x = vel_x
        twist.linear.y = vel_y
        twist.linear.z = vel_z
        twist.angular.z = 0 
        
        return twist

    def plot_data(self):
        rospy.loginfo("Generating diagram...")
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            
            # Global Title
            fig.suptitle(f"PID Controller Performance with Time Reference\n"
                         f"Gains: XY (Kp={self.kp_xy}, Kd={self.kd_xy}) | Z (Kp={self.kp_z}, Ki={self.ki_z}, Kd={self.kd_z})", 
                         fontsize=14, fontweight='bold')

            # --- XY Plot (Left) - NOW WITH COLORMAP ---
            # Using scatter instead of plot to visualize time
            sc = ax1.scatter(self.path_x, self.path_y, c=self.time_data, cmap='viridis', s=10, label='Actual Path')
            
            # Add a colorbar to indicate time
            cbar = plt.colorbar(sc, ax=ax1)
            cbar.set_label('Flight Time [s]')

            ax1.set_title("Trajectory Tracking: Square Path Manoeuvre", fontsize=12)
            ax1.set_xlabel("Position X [m]")
            ax1.set_ylabel("Position Y [m]")
            ax1.grid(True)
            ax1.axis('equal')
            
            # Reference Path (Black dashed line)
            route_x = [p[0] for p in self.route] + [self.route[0][0]]
            route_y = [p[1] for p in self.route] + [self.route[0][1]]
            ax1.plot(route_x, route_y, 'k--', alpha=0.5, label='Reference Path')
            
            # Start/End Points
            if self.path_x:
                ax1.scatter(self.path_x[0], self.path_y[0], color='green', s=100, label='Start Point', zorder=5)
                ax1.scatter(self.path_x[-1], self.path_y[-1], color='red', s=100, label='End Point', zorder=5)
            
            ax1.legend(loc='upper right', framealpha=0.9)

            # --- Z Plot (Right) ---
            ax2.plot(self.time_data, self.path_z, label='Measured Altitude', color='red', linewidth=1.5)
            ax2.set_title("Vertical Position Control Tracking", fontsize=12)
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Altitude [m]")
            ax2.axhline(y=4.0, color='green', linestyle='--', label='Setpoint (4m)', linewidth=2)
            ax2.grid(True)
            ax2.legend(loc='lower right', framealpha=0.9)

            # Save
            save_path = os.path.expanduser("~/flight_performance_analysis.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
            
            rospy.loginfo(f"SUCCESS: Diagram saved to: {save_path}")

        except Exception as e:
            rospy.logerr(f"Error plotting data: {e}")
            
    def print_final_stats(self):
        if not self.accuracy_stats:
            print("No statistics data collected.")
            return

        print("\n" + "="*40)
        print("FLIGHT ANALYSIS (Relative Accuracy)")
        print("="*40)
        
        total_rel_acc = 0
        max_error = 0
        
        for entry in self.accuracy_stats:
            idx = entry['index']
            err = entry['error_abs']
            rel = entry['rel_acc']
            print(f"WP {idx}: Dev. {err:.3f}m ({rel:.2%})")
            
            total_rel_acc += rel
            if err > max_error: max_error = err

        avg_rel_acc = total_rel_acc / len(self.accuracy_stats)
        
        print("-" * 40)
        print(f"AVERAGE (Relative): {avg_rel_acc:.4f} ({avg_rel_acc:.2%})")
        print(f"MAXIMUM ERROR (Absolute): {max_error:.3f} m")
        print("="*40 + "\n")

    def run(self):
        rate = rospy.Rate(30)
        rospy.sleep(1.0)
        self.last_time = rospy.Time.now()

        rospy.loginfo("Waiting for drone...")
        while not rospy.is_shutdown() and (self.current_state is None or not self.current_state.connected):
            rate.sleep()
        
        rospy.loginfo("Start!")
        try:
            while not rospy.is_shutdown():
                twist = self.compute_velocity()
                self.vel_pub.publish(twist)
                
                rate.sleep()
        except rospy.ROSInterruptException:
            pass 
        finally:
            self.print_final_stats()
            self.plot_data()

if __name__ == "__main__":
    controller = SurveillanceVelocity()
    spawn_quadrat_marker(controller.route, controller.altitude)
    spawn_colored_ground()
    
    # 3. Begin flight execution
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
