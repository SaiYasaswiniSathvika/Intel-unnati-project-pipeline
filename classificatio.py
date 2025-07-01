import cv2
import numpy as np
import openvino as ov

# Initialize OpenVINO Core
core = ov.Core()

# Load Detection Model
detection_model = core.read_model("intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml")
compiled_detection_model = core.compile_model(detection_model, "CPU")
input_layer_det = compiled_detection_model.input(0)
output_layer_det = compiled_detection_model.output(0)

# Load Vehicle Attribute Classification Model
vehicle_model = core.read_model("intel/vehicle-attributes-recognition-barrier-0039/FP16/vehicle-attributes-recognition-barrier-0039.xml")
compiled_vehicle_model = core.compile_model(vehicle_model, "CPU")
input_layer_vehicle = compiled_vehicle_model.input(0)
output_type = compiled_vehicle_model.output(0)
output_color = compiled_vehicle_model.output(1)

# Load Person Attribute Classification Model
person_model = core.read_model("intel/person-attributes-recognition-crossroad-0230/FP16/person-attributes-recognition-crossroad-0230.xml")
compiled_person_model = core.compile_model(person_model, "CPU")
person_input = compiled_person_model.input(0)
person_output = compiled_person_model.output(0)

# Label definitions
vehicle_types = ["car", "bus", "truck", "van"]
vehicle_colors = ["white", "gray", "yellow", "red", "green", "blue", "black"]
age_groups = ["<18", "18-25", "26-35", "36-45", "46-60", "60+"]

# Load Video
cap = cv2.VideoCapture("people_bicycle_car.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 30 != 0:
        continue

    print(f"\n[Frame {frame_count}] Raw Detections:")

    # Resize frame for detection input
    resized = cv2.resize(frame, (544, 320))
    input_image = np.expand_dims(np.transpose(resized, (2, 0, 1)), axis=0).astype(np.float32)

    # Run detection
    detection_result = compiled_detection_model([input_image])[output_layer_det][0][0]

    for det in detection_result:
        confidence = det[2]
        if confidence > 0.6:
            xmin = int(det[3] * frame.shape[1])
            ymin = int(det[4] * frame.shape[0])
            xmax = int(det[5] * frame.shape[1])
            ymax = int(det[6] * frame.shape[0])

            cropped = frame[ymin:ymax, xmin:xmax]
            if cropped.size == 0:
                continue

            box_h, box_w = cropped.shape[:2]

            if box_w > box_h:
                # Vehicle
                try:
                    vehicle_img = cv2.resize(cropped, (72, 72))
                    vehicle_input_blob = np.expand_dims(np.transpose(vehicle_img, (2, 0, 1)), axis=0).astype(np.float32)
                    results = compiled_vehicle_model([vehicle_input_blob])
                    type_idx = int(np.argmax(results[output_type]))
                    color_idx = int(np.argmax(results[output_color]))
                    if type_idx < len(vehicle_types) and color_idx < len(vehicle_colors):
                        print(f"🚗 Vehicle detected | Type: {vehicle_types[type_idx]}, Color: {vehicle_colors[color_idx]}")
                except Exception as e:
                    print(f"❌ Vehicle classification error: {e}")

            else:
                # Person
                try:
                    person_img = cv2.resize(cropped, (80, 160))  # Required input shape (W=80, H=160)
                    person_input_blob = np.expand_dims(np.transpose(person_img, (2, 0, 1)), axis=0).astype(np.float32)
                    result = compiled_person_model([person_input_blob])[person_output].flatten()
                    gender = "Male" if result[0] > 0.5 else "Female"
                    age_idx = int(np.argmax(result[2:6]))
                    age = age_groups[age_idx] if age_idx < len(age_groups) else "Unknown"
                    print(f"🧍 Person detected | Gender: {gender}, Age Group: {age}")
                except Exception as e:
                    print(f"❌ Person classification error: {e}")

cap.release()
print("\n✅ Person + Vehicle classification complete.")
