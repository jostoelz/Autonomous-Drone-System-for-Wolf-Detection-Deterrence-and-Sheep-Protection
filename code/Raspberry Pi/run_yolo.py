import rospy
import cv2
import numpy as np
import threading
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# --- CONFIGURATION ---
HEF_FILE = "yolov8s.hef"
TARGET_CLASS_ID = 2        # ID 2 = Car (Coco standard)
CONFIDENCE_THRESHOLD = 0.3 # Sensitivity

# --- FAST IMAGE PROCESSING ---
def letterbox_image_cv2(image, target_width, target_height):
    """
    Resizes image to model input size using OpenCV (C++ acceleration).
    Much faster than PIL.
    """
    ih, iw = image.shape[:2]
    w, h = target_width, target_height
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Fast linear interpolation
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Create gray background
    new_image = np.full((h, w, 3), 114, dtype=np.uint8)

    # Paste centered
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy:dy+nh, dx:dx+nw] = image_resized

    return new_image, (scale, dx, dy)

class RealTimeDetector:
    def __init__(self):
        self.latest_msg = None
        self.new_data = False
        self.lock = threading.Lock() 
        
        # Publisher: queue_size=1 ensures we don't send old detections
        self.pub = rospy.Publisher('/object_detection/target', Point, queue_size=1)
        
        # Subscriber: queue_size=3 acts as a small buffer against Wi-Fi jitter.
        self.subscriber = rospy.Subscriber("/raspicam_node/image/compressed", 
                                           CompressedImage, self.callback, queue_size=3, buff_size=2**24)

    def callback(self, msg):
        # Extremely lightweight callback
        with self.lock:
            self.latest_msg = msg
            self.new_data = True

    def get_latest_image(self):
        msg = None
        with self.lock:
            if self.new_data:
                msg = self.latest_msg
                self.new_data = False
        return msg

    def process_loop(self, infer_pipeline, model_w, model_h):
        """
        Main Loop. Runs synchronous to the camera input.
        """
        frame_count = 0
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            # 1. Get Image
            msg = self.get_latest_image()
            
            if msg is None:
                # Sleep extremely briefly to yield CPU
                rospy.sleep(0.001) 
                continue

            try:
                # 2. Decode (Numpy -> OpenCV)
                # This takes ~1ms for 320x240 images
                np_arr = np.frombuffer(msg.data, np.uint8)
                image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if image_cv is None: continue

                # OpenCV BGR -> RGB
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                # 3. Resize / Letterbox
                # This takes ~1.5ms
                processed_image, (scale, pad_x, pad_y) = letterbox_image_cv2(image_rgb, model_w, model_h)
                input_data = np.expand_dims(processed_image, axis=0)

                # 4. AI Inference
                # This takes ~14ms
                infer_results = infer_pipeline.infer(input_data)
                
                key = list(infer_results.keys())[0]
                detections = infer_results[key][0]

                # 5. Extract Best Target
                best_score = -1.0
                best_det = None

                if len(detections) > TARGET_CLASS_ID:
                    class_dets = detections[TARGET_CLASS_ID]
                    for det in class_dets:
                        ymin, xmin, ymax, xmax, score = det
                        if score > best_score and score > CONFIDENCE_THRESHOLD:
                            best_score = score
                            best_det = (ymin, xmin, ymax, xmax)

                # 6. Publish
                if best_det:
                    ymin, xmin, ymax, xmax = best_det
                    orig_x = (xmin * model_w - pad_x) / scale
                    orig_y = (ymin * model_h - pad_y) / scale
                    
                    h_orig, w_orig = image_cv.shape[:2]
                    orig_x = max(0, min(orig_x, w_orig))
                    orig_y = max(0, min(orig_y, h_orig))

                    p = Point()
                    p.x = float(orig_x)
                    p.y = float(orig_y)
                    p.z = 0.0
                    self.pub.publish(p)

                # 7. FPS Counter (Every 100 frames)
                frame_count += 1
                if frame_count % 100 == 0:
                    curr_time = rospy.get_time()
                    elapsed = curr_time - start_time
                    fps = 100 / elapsed
                    rospy.loginfo(f"Performance: {fps:.1f} FPS (Last 100 frames)")
                    start_time = curr_time
                    frame_count = 0
                
            except Exception as e:
                rospy.logerr_throttle(5, f"Error: {e}")

def main():
    rospy.init_node('hailo_yolo_final', anonymous=True)
    detector = RealTimeDetector()
    
    # --- HAILO SETUP ---
    hef = HEF(HEF_FILE)

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        input_info = hef.get_input_vstream_infos()[0]
        model_h, model_w = input_info.shape[0], input_info.shape[1]

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                detector.process_loop(infer_pipeline, model_w, model_h)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
