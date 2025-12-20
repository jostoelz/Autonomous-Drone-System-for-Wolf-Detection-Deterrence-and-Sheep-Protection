import rospy
import cv2
import numpy as np
import threading
import time
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# --- CONFIGURATION ---
HEF_YOLO = "yolov8s.hef"
HEF_MIDAS = "midas_v2_small_hailo8_256_320.hef"

TARGET_CLASS_ID = 2        # ID 2 = Car (COCO dataset standard)
CONFIDENCE_THRESHOLD = 0.3 # Sensitivity cutoff
CALIBRATION_FACTOR = 10.0  # Needs manual calibration

class FusionNode:
    def __init__(self):
        self.latest_msg = None
        self.new_data = False
        self.lock = threading.Lock() 
        
        # ROS Publishers
        self.pub_target = rospy.Publisher('/object_detection/target', Point, queue_size=1)
        self.pub_dist = rospy.Publisher('/distance/target', Float32, queue_size=1)
        
        # Subscribe to compressed images to save bandwidth
        self.subscriber = rospy.Subscriber("/raspicam_node/image/compressed", 
                                           CompressedImage, self.callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("Fusion Node (Sequential Mode) initialized.")

    def callback(self, msg):
        # Simple callback: just grab the latest frame and drop the old ones to minimize latency
        with self.lock:
            self.latest_msg = msg
            self.new_data = True

    def get_latest_image(self):
        # Thread-safe retrieval of the input image
        msg = None
        with self.lock:
            if self.new_data:
                msg = self.latest_msg
                self.new_data = False
        return msg

    def preprocess_yolo(self, image, target_w, target_h):
        # Standard "Letterbox" resizing
        # Keeps aspect ratio by adding gray padding, which is better for detection accuracy
        ih, iw = image.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Fill background with neutral gray (114)
        new_image = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dx, dy = (target_w - nw) // 2, (target_h - nh) // 2
        new_image[dy:dy+nh, dx:dx+nw] = image_resized
        
        return new_image, scale, dx, dy

    def preprocess_midas(self, image, target_w, target_h):
        # MiDaS is pretty robust to distortion, so a simple stretch resize is faster and fine
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def process_loop(self, net_group_yolo, params_yolo, inp_yolo, outp_yolo, yolo_dims,
                           net_group_midas, params_midas, inp_midas, outp_midas, midas_dims):
        
        yolo_h, yolo_w = yolo_dims
        midas_h, midas_w = midas_dims
        
        frame_count = 0
        start_time = rospy.get_time()

        rospy.loginfo("Starting Sequential Loop...")

        while not rospy.is_shutdown():
            msg = self.get_latest_image()
            if msg is None:
                rospy.sleep(0.001) # Sleep briefly to yield CPU
                continue

            try:
                # Decode the compressed ROS message to OpenCV
                np_arr = np.frombuffer(msg.data, np.uint8)
                image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image_cv is None: continue
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = image_rgb.shape[:2]

                # STEP 1: YOLO INFERENCE
                # Performs a "manual context switch" here
                # Activate -> Infer -> Deactivate (automatically handled by 'with')
              
                best_det = None
                
                with net_group_yolo.activate(params_yolo):
                    with InferVStreams(net_group_yolo, inp_yolo, outp_yolo) as pipe_yolo:
                        
                        # Preprocess and infer
                        img_yolo, scale, pad_x, pad_y = self.preprocess_yolo(image_rgb, yolo_w, yolo_h)
                        input_yolo = np.expand_dims(img_yolo, axis=0)
                        
                        res_yolo = pipe_yolo.infer(input_yolo)
                        dets = res_yolo[list(res_yolo.keys())[0]][0]

                        # Parse results to find the best target
                        best_score = -1.0
                        if len(dets) > TARGET_CLASS_ID:
                            for det in dets[TARGET_CLASS_ID]:
                                ymin, xmin, ymax, xmax, score = det
                                if score > best_score and score > CONFIDENCE_THRESHOLD:
                                    best_score = score
                                    best_det = (ymin, xmin, ymax, xmax)
                
                # At this point, YOLO is deactivated and the hardware is free

                # STEP 2: DEPTH ESTIMATION
                # Only run this if we actually found something to measure
                dist_val = 0.0
                target_found = False

                if best_det:
                    target_found = True
                    ymin, xmin, ymax, xmax = best_det
                    
                    # 1. Calculate center in YOLO coordinates (Normalized 0.0 - 1.0)
                    cx_norm = (xmin + xmax) / 2.0
                    cy_norm = (ymin + ymax) / 2.0

                    # 2. Transform YOLO coords -> Original Image Pixels
                    # We reverse the padding/letterboxing here
                    orig_x = (cx_norm * yolo_w - pad_x) / scale
                    orig_y = (cy_norm * yolo_h - pad_y) / scale
                    
                    # Clip to be safe (ensure we are within the camera frame)
                    orig_x = max(0, min(orig_x, orig_w - 1))
                    orig_y = max(0, min(orig_y, orig_h - 1))

                    # Publish the Point for the robot
                    p = Point()
                    p.x = float(orig_x)
                    p.y = float(orig_y)
                    self.pub_target.publish(p)

                    with net_group_midas.activate(params_midas):
                        with InferVStreams(net_group_midas, inp_midas, outp_midas) as pipe_midas:
                            
                            img_midas = self.preprocess_midas(image_rgb, midas_w, midas_h)
                            input_midas = np.expand_dims(img_midas, axis=0)
                            
                            res_midas = pipe_midas.infer(input_midas)
                            raw_depth_map = res_midas[list(res_midas.keys())[0]][0]
                            if raw_depth_map.ndim == 3: raw_depth_map = raw_depth_map[:, :, 0]

                            # 3. Transform Original Image Pixels -> MiDaS Coordinates
                            # Since MiDaS is STRETCHED (no padding), we simply scale using the ratio
                            # logic: midas_x = orig_x * (midas_width / original_width)
                            mx = int(orig_x * (midas_w / orig_w))
                            my = int(orig_y * (midas_h / orig_h))
                            
                            # Safety clip
                            mx = max(0, min(mx, midas_w - 1))
                            my = max(0, min(my, midas_h - 1))

                            raw_val = raw_depth_map[my, mx]
                            dist_val = CALIBRATION_FACTOR / raw_val if raw_val > 1e-4 else 99.9
                            
                            self.pub_dist.publish(dist_val)
                
                # MiDaS is now deactivated.

                # Simple performance logging
                frame_count += 1
                if frame_count % 10 == 0: # Log more often since FPS will be lower due to switching
                    elapsed = rospy.get_time() - start_time
                    fps = 10 / elapsed
                    start_time = rospy.get_time()
                    log_msg = f"FPS: {fps:.1f}"
                    if target_found:
                        log_msg += f" | Dist: {dist_val:.2f}m"
                    rospy.loginfo(log_msg)

            except Exception as e:
                rospy.logerr_throttle(5, f"Error: {e}")

def main():
    rospy.init_node('hailo_fusion_node', anonymous=True)
    node = FusionNode()
    
    # 1. Load both compiled models (HEFs)
    hef1 = HEF(HEF_YOLO)
    hef2 = HEF(HEF_MIDAS)

    # Use basic VDevice (without params) to avoid compatibility issues on RPi
    with VDevice() as target:
        
        # 2. Configure both networks
        # This loads the weights into the chip's memory but doesn't start execution yet
        conf1 = ConfigureParams.create_from_hef(hef1, interface=HailoStreamInterface.PCIe)
        net_group1 = target.configure(hef1, conf1)[0]
        params1 = net_group1.create_params()
        
        conf2 = ConfigureParams.create_from_hef(hef2, interface=HailoStreamInterface.PCIe)
        net_group2 = target.configure(hef2, conf2)[0]
        params2 = net_group2.create_params()

        # 3. Prepare stream parameters
        in_p1 = InputVStreamParams.make(net_group1, format_type=FormatType.UINT8)
        out_p1 = OutputVStreamParams.make(net_group1, format_type=FormatType.FLOAT32)
        
        in_p2 = InputVStreamParams.make(net_group2, format_type=FormatType.UINT8)
        out_p2 = OutputVStreamParams.make(net_group2, format_type=FormatType.FLOAT32)

        # Get input dimensions dynamically from the loaded HEF
        info1 = hef1.get_input_vstream_infos()[0]
        yolo_dims = (info1.shape[0], info1.shape[1])
        
        info2 = hef2.get_input_vstream_infos()[0]
        midas_dims = (info2.shape[0], info2.shape[1])
        
        # 4. Start the loop
        # Pass the network groups and params so we can activate them inside the loop
        node.process_loop(net_group1, params1, in_p1, out_p1, yolo_dims,
                          net_group2, params2, in_p2, out_p2, midas_dims)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
