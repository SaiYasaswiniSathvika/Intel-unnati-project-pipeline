import time
import cv2
import numpy as np
import openvino as ov

def run_single_stream():
    core = ov.Core()
    model = core.read_model("intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml")
    compiled_model = core.compile_model(model, "CPU")
    
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    cap = cv2.VideoCapture("pedestrian.avi")
    
    frame_count = 0
    total_latency = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        resized = cv2.resize(frame, (544, 320))
        input_tensor = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0).astype(np.float32) / 255.0

        # Inference with latency measurement
        start = time.time()
        result = compiled_model([input_tensor])[output_layer]
        latency = (time.time() - start) * 1000
        
        # Update statistics
        frame_count += 1
        total_latency += latency

        # Periodic reporting (same format as multi-stream)
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            avg_latency = total_latency / frame_count
            print(f"FPS: {fps:.2f}, Avg Latency: {avg_latency:.2f} ms")  # Omitting [Stream 1] for single stream

    cap.release()
    final_fps = frame_count / (time.time() - start_time)
    print(f"Final FPS: {final_fps:.2f}, Total Frames: {frame_count}")

if __name__ == "__main__":
    run_single_stream()