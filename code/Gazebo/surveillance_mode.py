#!/usr/bin/env python3
import rospy
import os 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from geometry_msgs.msg import Point, PoseStamped

class SurveillanceVelocity:
    def __init__(self):
        rospy.init_node("surveillance_node")

        self.vel_pub = rospy.Publisher("/surveillance_velocity", Twist, queue_size=10)

        rospy.Subscriber("/mavros/state", State, self.state_cb)
        self.current_state = State()

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)

        self.route = [(0, 0), (10, 0), (10,10), (0, 10)]
        self.current_target_index = 0

        self.current_position = Point()
        self.current_position.z = 0.0
        self.altitude = 4.0
        self.max_vel = 3.0

        # --- TUNING AREA ---
        
        # X/Y: PERFECT (Do not change!)
        self.kp_xy = 1.2   
        self.kd_xy = 0.6   
        self.tolerance = 0.3 
        
        # Z-Axis (Altitude): NEW TUNING
        self.kp_z = 1.5    # P: A bit more power for the ascent
        self.ki_z = 0.2    # I: Small, only to counteract gravity offset (prevents oscillations)
        self.kd_z = 0.8    # D: Brake loosened (was 1.8, which was too much)
        
        # Storage
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_dz = 0.0
        
        self.integral_z = 0.0
        self.last_time = rospy.Time.now()
        self.first_run = True
        
        # Plotting Arrays
        self.path_x = []
        self.path_y = []
        self.path_z = []
        self.time_data = []
        self.start_time = rospy.Time.now().to_sec()
        self.accuracy_stats = []  # Here we store every measurement
        self.last_waypoint = (0, 0) # Start position (first segment starts here)

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_position = msg.pose.position
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
            # --- NEW: Accuracy Calculation ---
            # 1. Calculate length of the just flown segment (x)
            prev = self.last_waypoint
            curr = self.route[self.current_target_index]
            
            # Pythagorean theorem for segment length
            segment_length = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
            
            # Protection against division by zero (at start)
            if segment_length > 0:
                # relative accuracy = error / distance
                rel_accuracy = distance / segment_length
                self.accuracy_stats.append({
                    'index': self.current_target_index,
                    'error_abs': distance,
                    'dist_x': segment_length,
                    'rel_acc': rel_accuracy
                })
                
                rospy.loginfo(f"STATISTICS: Target {self.current_target_index} reached.")
                rospy.loginfo(f"  -> Error absolute: {distance:.3f} m")
                rospy.loginfo(f"  -> On distance:    {segment_length:.1f} m")
                rospy.loginfo(f"  -> Rel. Accuracy:  {rel_accuracy:.2%} (Error per meter)")

            # Update for the next segment
            self.last_waypoint = curr
            # ------------------------------------

            self.current_target_index = (self.current_target_index + 1) % len(self.route)
            # ... (your code for target update) ...
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
        
        # I-Term (Standard integration with limit)
        # We removed the "if abs(dz) < 1.0" because it caused jumps
        self.integral_z += dz * dt
        
        # Limit (Anti-Windup) to 2.0 (easily enough for the gravity offset of ~0.25)
        self.integral_z = max(min(self.integral_z, 2.0), -2.0) 
        
        # D-Term Z
        if self.first_run:
            d_dz = 0.0
            self.first_run = False
        else:
            d_dz = (dz - self.last_dz) / dt
        self.last_dz = dz

        # PID Formula Z
        p_term = self.kp_z * dz
        i_term = self.ki_z * self.integral_z
        d_term = self.kd_z * d_dz
        
        vel_z = p_term + i_term + d_term
        
        # Limits (Max Speed)
        vel_x = max(min(vel_x, self.max_vel), -self.max_vel)
        vel_y = max(min(vel_y, self.max_vel), -self.max_vel)
        vel_z = max(min(vel_z, self.max_vel), -self.max_vel)
        
        rospy.loginfo_throttle(0.5, f"Z:{self.current_position.z:.2f} | P:{p_term:.2f} I:{i_term:.2f} D:{d_term:.2f} | Cmd:{vel_z:.2f}")

        twist = Twist()
        twist.linear.x = vel_x
        twist.linear.y = vel_y
        twist.linear.z = vel_z
        twist.angular.z = 0 
        
        return twist

    def plot_data(self):
        rospy.loginfo("Generating diagram...")
        try:
            # "constrained_layout=True" often helps better with long titles than tight_layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

            # --- XY Plot (Left) ---
            ax1.plot(self.path_x, self.path_y, label='Actual Trajectory', color='blue', linewidth=2)
            
            # Title with more context
            ax1.set_title("Trajectory Tracking: Square Path Manoeuvre", fontsize=12)
            ax1.set_xlabel("Position X [m]")
            ax1.set_ylabel("Position Y [m]")
            ax1.grid(True)
            ax1.axis('equal')
            
            # Reference path
            route_x = [p[0] for p in self.route] + [self.route[0][0]]
            route_y = [p[1] for p in self.route] + [self.route[0][1]]
            ax1.plot(route_x, route_y, 'k--', alpha=0.5, label='Reference Path')
            
            # Start/End
            if self.path_x:
                ax1.plot(self.path_x[0], self.path_y[0], 'go', label='Start Point')
                ax1.plot(self.path_x[-1], self.path_y[-1], 'ro', label='End Point')
            
            ax1.legend(loc='upper right', framealpha=0.9)

            # --- Z Plot (Right) ---
            ax2.plot(self.time_data, self.path_z, label='Measured Altitude', color='red', linewidth=1.5)
            
            # Title with more context
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
    controller.run()
