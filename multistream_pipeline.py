import time
import cv2
import numpy as np
import threading
import openvino as ov

# Load model only once
core = ov.Core()
model_path = "intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml"
model = core.read_model(model_path)

def process_stream(video_path, stream_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Stream {stream_id}]  Unable to open video: {video_path}")
        return

    #  Compile model inside thread (unique to each thread)
    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    frame_count = 0
    total_latency = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (544, 320))
        input_image = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0).astype(np.float32) / 255.0

        start = time.time()
        result = compiled_model([input_image])[output_layer]
        end = time.time()

        latency = (end - start) * 1000
        total_latency += latency
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            avg_latency = total_latency / frame_count
            print(f"[Stream {stream_id}] FPS: {fps:.2f}, Avg Latency: {avg_latency:.2f} ms")

    cap.release()
    total_time = time.time() - start_time
    final_fps = frame_count / total_time if total_time > 0 else 0
    print(f"[Stream {stream_id}]  Final FPS: {final_fps:.2f}, Total Frames: {frame_count}")

# List of test videos (ensure all are valid and readable)
video_paths = [
    "pedestrian.avi",
    "people_bicycle_car.mp4",
    "pedestrian.avi",
    "people_bicycle_car.mp4",
    "pedestrian.avi",
    "people_bicycle_car.mp4"
]

#  Run all streams
threads = []
for idx, path in enumerate(video_paths):
    t = threading.Thread(target=process_stream, args=(path, idx + 1))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Multi-stream inference complete.")
