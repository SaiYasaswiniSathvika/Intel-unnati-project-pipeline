import cv2
import numpy as np
import openvino as ov

core = ov.Core()

# Load detection model (same)
detection_model = core.read_model("intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml")
compiled_detection_model = core.compile_model(detection_model, "CPU")
input_layer_det = compiled_detection_model.input(0)
output_layer_det = compiled_detection_model.output(0)

# Load vehicle attributes model
vehicle_model = core.read_model("intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml")
compiled_vehicle_model = core.compile_model(vehicle_model, "CPU")
output_type = compiled_vehicle_model.output(0)
output_color = compiled_vehicle_model.output(1)

vehicle_types = ["car", "bus", "truck", "van"]
vehicle_colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]

cap = cv2.VideoCapture("people_bicycle_car.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 != 0:
        continue

    resized = cv2.resize(frame, (544, 320))
    input_image = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0).astype(np.float32)
    detections = compiled_detection_model([input_image])[output_layer_det][0][0]

    print(f"\n[Frame {frame_count}] Vehicle Detections:")
    for det in detections:
        if det[2] > 0.6:
            xmin = int(det[3] * frame.shape[1])
            ymin = int(det[4] * frame.shape[0])
            xmax = int(det[5] * frame.shape[1])
            ymax = int(det[6] * frame.shape[0])

            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            try:
                resized_crop = cv2.resize(crop, (72, 72))
                blob = np.expand_dims(np.transpose(resized_crop, (2, 0, 1)), axis=0).astype(np.float32)
                results = compiled_vehicle_model([blob])
                type_idx = int(np.argmax(results[output_type]))
                color_idx = int(np.argmax(results[output_color]))

                if type_idx < len(vehicle_types) and color_idx < len(vehicle_colors):
                    print(f" Vehicle Type: {vehicle_types[type_idx]}, Color: {vehicle_colors[color_idx]}")
                else:
                    print(" Invalid label index.")
            except Exception as e:
                print(" Error in vehicle classification:", e)

cap.release()
print("\n Vehicle classification complete.")
