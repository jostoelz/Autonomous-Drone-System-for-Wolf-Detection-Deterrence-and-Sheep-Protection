import sys
import gi
import time
import os

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# --- CONFIGURATION ---
HEF_PATH = "yolov8s.hef"
VIDEO_PATH = "Wolf.mp4"
CONFIG_PATH = "yolo_config.json"
SO_PATH = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"

MODEL_RES = 640

# Globals for performance measurement
frame_count = 0
window_start_time = 0   # Timer for the last 30 frames
total_start_time = 0    # Timer for the entire duration

def run_pipeline():
    Gst.init(None)

    if not os.path.exists(HEF_PATH) or not os.path.exists(CONFIG_PATH) or not os.path.exists(SO_PATH):
        print("Error: Files missing!")
        return

    # --- PIPELINE ---
    pipeline_str = (
        f"filesrc location={VIDEO_PATH} ! "
        "qtdemux ! h264parse ! avdec_h264 ! "
        "videoconvert ! "
        f"videoscale method=0 add-borders=true ! "
        f"video/x-raw, width={MODEL_RES}, height={MODEL_RES}, format=RGB, pixel-aspect-ratio=1/1 ! "
        f"hailonet name=hailo_infer hef-path={HEF_PATH} ! "
        f"hailofilter so-path={SO_PATH} config-path={CONFIG_PATH} qos=false ! "
        "hailooverlay ! "
        "videoconvert ! "
        "fpsdisplaysink video-sink=autovideosink text-overlay=true sync=false"
    )

    print(f"Starting pipeline...")
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"GStreamer Error: {e}")
        return

    # --- PERFORMANCE PROBE ---
    hailo_element = pipeline.get_by_name("hailo_infer")
    if hailo_element:
        src_pad = hailo_element.get_static_pad("src")
        src_pad.add_probe(Gst.PadProbeType.BUFFER, fps_probe_callback)
    else:
        print("Warning: Could not attach performance probe.")

    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    try:
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_message, loop)
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)

def fps_probe_callback(pad, info):
    """Calculates both current (window) FPS and total average FPS."""
    global frame_count, window_start_time, total_start_time
    
    current_time = time.time()

    # Initialize timers on the very first frame
    if frame_count == 0:
        total_start_time = current_time
        window_start_time = current_time
    
    frame_count += 1
    
    # Update display every 30 frames
    if frame_count % 30 == 0:
        # 1. Calculate Current FPS (last 30 frames)
        window_duration = current_time - window_start_time
        current_fps = 30 / window_duration if window_duration > 0 else 0
        
        # 2. Calculate Total Average FPS (since start)
        total_duration = current_time - total_start_time
        avg_fps = frame_count / total_duration if total_duration > 0 else 0
        avg_latency = (1.0 / avg_fps) * 1000 if avg_fps > 0 else 0

        print(f"Current: {current_fps:.1f} FPS | AVERAGE: {avg_fps:.1f} FPS ({avg_latency:.1f} ms)")
        
        # Reset window timer (but NOT the total timer)
        window_start_time = current_time
        
    return Gst.PadProbeReturn.OK

def on_message(bus, message, loop):
    mtype = message.type
    if mtype == Gst.MessageType.EOS:
        print("\n--- Summary ---")
        # Final calculation at the end
        if total_start_time > 0:
            total_duration = time.time() - total_start_time
            final_fps = frame_count / total_duration
            print(f"Total Frames: {frame_count}")
            print(f"Total Time:   {total_duration:.2f} s")
            print(f"Final Avg FPS: {final_fps:.2f}")
        loop.quit()
    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    return True

if __name__ == "__main__":
    run_pipeline()
