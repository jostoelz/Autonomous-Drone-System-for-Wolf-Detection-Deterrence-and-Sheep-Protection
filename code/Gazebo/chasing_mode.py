#!/usr/bin/env python3
import rospy
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from mavros_msgs.msg import State
from geometry_msgs.msg import Point, PoseStamped, Twist
from std_msgs.msg import Float32

class ChasingVelocity:
    def __init__(self):
        rospy.init_node("chasing_node")

        self.vel_pub = rospy.Publisher("/chasing_velocity", Twist, queue_size=10)

        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)
        
        self.current_position = Point()
        self.current_position.z = 0.0

        # Sensor subscriptions
        rospy.Subscriber("/normalized_bounding_box", Point, self.normalized_bounding_box_cb)
        rospy.Subscriber("/depth", Float32, self.depth_cb)
        
        self.normalized_bounding_box = Point()
        self.normalized_bounding_box.x = 0.0
        self.last_box_time = rospy.Time.now()
        
        # Variables for sensor data filtering
        self.depth = Float32()
        self.depth.data = 2.0 
        self.last_valid_depth = 2.0 # Buffer to store the last reliable depth reading

        # Flight parameters
        self.altitude = 1.0        
        self.min_distance_object = 1.5 
        self.max_vel = 3.0
        self.max_yaw_rate = 2.5 # Increased limit to allow faster rotation

        # --- TUNING PARAMETERS ---
        
        # 1. Yaw Control: Increased gain for faster alignment with the target
        self.kp_yaw = 1.2  
        
        # 2. Distance Control (Linear X)
        self.kp_x = 1.0

        # 3. Altitude Control (Linear Z): Softer gains to prevent altitude drops during rapid moves
        self.kp_z = 1.0    
        self.ki_z = 0.1    
        self.kd_z = 0.4    
        # Internal storage for PID logic
        self.integral_z = 0.0
        self.last_dz = 0.0
        self.last_time = rospy.Time.now()
        self.first_run = True

        # --- DATA LOGGING INITIALIZATION ---
        self.start_time = rospy.Time.now().to_sec()
        self.data_time = []
        self.data_x = []
        self.data_y = []
        self.data_visual_error = []
        self.data_depth_meas = []
        self.data_depth_ref = []
        self.data_cmd_vel_x = []
        self.data_cmd_vel_z = []

        # Define the route of the target box for visualization purposes
        # Note: These coordinates match the animation defined in the C++ simulation plugin
        self.box_route = [(50, 0), (0, 0), (10, 10), (10, 20), (20, 10)]

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_position = msg.pose.position
        self.data_x.append(msg.pose.position.x)
        self.data_y.append(msg.pose.position.y)

    def normalized_bounding_box_cb(self, msg):
        self.normalized_bounding_box = msg
        self.last_box_time = rospy.Time.now()

    def depth_cb(self, msg):
        self.depth = msg

    def compute_velocity(self):
       now = rospy.Time.now()
       dt = (now - self.last_time).to_sec()
       self.last_time = now

       if dt <= 0 or dt > 1.0: dt = 0.033
       
       # Calculate current time relative to start for plotting
       current_t = rospy.Time.now().to_sec() - self.start_time
       self.data_time.append(current_t)

       # --- TIMEOUT CHECK ---
       time_diff = (rospy.Time.now() - self.last_box_time).to_sec()
       if time_diff > 0.5:
           dy = 0.0
           object_visible = False
       else:
           dy = self.normalized_bounding_box.x
           object_visible = True

       self.data_visual_error.append(dy)

       # --- SENSOR FILTERING (Spike Removal) ---
       raw_depth = self.depth.data
       # Filter: Reject illogical values (<0.1, >10) or large jumps (>0.8m)
       if raw_depth < 0.1 or raw_depth > 10.0:
           filtered_depth = self.last_valid_depth
       elif abs(raw_depth - self.last_valid_depth) > 0.8:
           # Assumption: Target lost due to a sharp corner; hold last valid value momentarily
           filtered_depth = self.last_valid_depth
       else:
           filtered_depth = raw_depth
           self.last_valid_depth = raw_depth

       self.data_depth_meas.append(raw_depth) # Log raw data to visualize filter performance
       self.data_depth_ref.append(self.min_distance_object)

       # --- CONTROL LOOP ---
       dx = filtered_depth - self.min_distance_object 
       dz = self.altitude - self.current_position.z    

       # Altitude Control (PID Implementation)
       self.integral_z += dz * dt
       self.integral_z = max(min(self.integral_z, 2.0), -2.0) 
       
       if self.first_run:
            d_dz = 0.0
            self.first_run = False
       else:
            d_dz = (dz - self.last_dz) / dt
       self.last_dz = dz

       vel_z = (self.kp_z * dz) + (self.ki_z * self.integral_z) + (self.kd_z * d_dz)
       vel_z = max(min(vel_z, 1.5), -1.0) # Clamp output to prevent excessive descent rates

       # Yaw Control
       # Aggressive P-controller to keep the target centered
       yaw_cmd = max(min(-dy * self.kp_yaw, self.max_yaw_rate), -self.max_yaw_rate)

       # Forward Control (Linear X) with "Corner Braking"
       # Reduce forward speed when the lateral error (dy) is high
       speed_factor = max(0.2, 1.0 - abs(dy) * 2.0) # Speed drops to 0.2 if error is 0.4 or higher
       cmd_vel_x = max(min(dx * self.kp_x, self.max_vel), -self.max_vel)
       cmd_vel_x = cmd_vel_x * speed_factor

       # Log control outputs
       self.data_cmd_vel_x.append(cmd_vel_x)
       self.data_cmd_vel_z.append(vel_z)

       twist = Twist()
       twist.linear.x = cmd_vel_x
       twist.linear.y = 0
       twist.linear.z = vel_z
       twist.angular.x = 0
       twist.angular.y = 0
       twist.angular.z = yaw_cmd
       
       rospy.loginfo_throttle(0.5, f"Vis:{object_visible} | Dist:{filtered_depth:.2f} | SpeedFactor:{speed_factor:.2f}")
       
       return twist

    def plot_dashboard(self):
        rospy.loginfo("Generating scientific plots...")
        
        # Synchronize array lengths to prevent plotting errors
        min_len = min(len(self.data_time), len(self.data_visual_error), len(self.data_depth_meas), len(self.data_x))
        t = self.data_time[:min_len]
        
        # Update plot styling for scientific publication standards
        plt.rcParams.update({'font.size': 10, 'font.family': 'serif'}) 
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8)) 
        
        # --- 1. VISUAL TRACKING ERROR (Top Left) ---
        axs[0, 0].plot(t, self.data_visual_error[:min_len], 'b-', linewidth=1.2, label='Tracking Error $e_u$')
        axs[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
        # Highlight optimal range
        axs[0, 0].fill_between(t, -0.1, 0.1, color='green', alpha=0.1, label='Deadband ($\pm 10\%$)')
        
        axs[0, 0].set_title('Lateral Visual Tracking Error', fontsize=11)
        axs[0, 0].set_ylabel('Norm. Horizontal Error $e_u$ [-]')
        axs[0, 0].set_xlabel('Time $t$ [s]')
        axs[0, 0].set_ylim(-1.0, 1.0)
        axs[0, 0].grid(True, linestyle=':', alpha=0.6)
        axs[0, 0].legend(loc='upper right', fontsize=8)

        # --- 2. DISTANCE REGULATION (Top Right) ---
        axs[0, 1].plot(t, self.data_depth_meas[:min_len], 'r-', linewidth=1.2, label='Measured $d_{meas}$')
        axs[0, 1].plot(t, self.data_depth_ref[:min_len], 'g--', linewidth=1.5, label='Setpoint $d_{ref}$')
        
        axs[0, 1].set_title('Longitudinal Distance Regulation', fontsize=11)
        axs[0, 1].set_ylabel('Relative Distance $d$ [m]')
        axs[0, 1].set_xlabel('Time $t$ [s]')
        axs[0, 1].grid(True, linestyle=':', alpha=0.6)
        axs[0, 1].legend(loc='upper right', fontsize=8)

        # --- 3. TRAJECTORY TRACKING (Bottom Left) ---
        # Plot target path (ground truth)
        box_x = [p[0] for p in self.box_route]
        box_y = [p[1] for p in self.box_route]
        axs[1, 0].plot(box_x, box_y, 'k--', linewidth=1.5, alpha=0.6, label='Target Path (Ground Truth)')
        
        # Plot UAV path with time colormap
        sc = axs[1, 0].scatter(self.data_x[:min_len], self.data_y[:min_len], c=t, cmap='viridis', s=10, label='UAV Trajectory')
        
        # Configure colorbar
        cbar = plt.colorbar(sc, ax=axs[1, 0])
        cbar.set_label('Time $t$ [s]', fontsize=9)
        
        axs[1, 0].set_title('2D Trajectory Tracking (Inertial Frame)', fontsize=11)
        axs[1, 0].set_xlabel('Position $X_I$ [m]')
        axs[1, 0].set_ylabel('Position $Y_I$ [m]')
        axs[1, 0].axis('equal')
        axs[1, 0].grid(True, linestyle=':', alpha=0.6)
        axs[1, 0].legend(loc='upper right', fontsize=8)

        # --- 4. CONTROLLER OUTPUTS (Bottom Right) ---
        axs[1, 1].plot(t, self.data_cmd_vel_x[:min_len], 'm-', linewidth=1.2, alpha=0.8, label='$v_{x,cmd}$ (Forward)')
        axs[1, 1].plot(t, self.data_cmd_vel_z[:min_len], 'c-', linewidth=1.2, alpha=0.8, label='$v_{z,cmd}$ (Altitude)')
        
        axs[1, 1].set_title('Controller Output Velocities', fontsize=11)
        axs[1, 1].set_ylabel('Linear Velocity Command [m/s]')
        axs[1, 1].set_xlabel('Time $t$ [s]')
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)
        axs[1, 1].legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        
        # Save figure to file
        save_path = os.path.expanduser("~/scientific_chasing_results.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        rospy.loginfo(f"Scientific plot saved to: {save_path}")

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and not self.current_state.connected:
            rate.sleep()
        self.last_time = rospy.Time.now()
        try:
            while not rospy.is_shutdown():
                twist = self.compute_velocity()
                self.vel_pub.publish(twist)
                rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.plot_dashboard()

if __name__ == "__main__":
    controller = ChasingVelocity()
    controller.run()
