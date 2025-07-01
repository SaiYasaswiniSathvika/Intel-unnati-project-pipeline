import cv2
import numpy as np
import openvino as ov

core = ov.Core()

# Load detection model
detection_model = core.read_model("intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml")
compiled_detection_model = core.compile_model(detection_model, "CPU")
input_layer_det = compiled_detection_model.input(0)
output_layer_det = compiled_detection_model.output(0)

# Load person attributes model
person_model = core.read_model("intel/person-attributes-recognition-crossroad-0230/FP16/person-attributes-recognition-crossroad-0230.xml")
compiled_person_model = core.compile_model(person_model, "CPU")
person_output = compiled_person_model.output(0)

age_groups = ["<18", "18-25", "26-35", "36-45", "46-60", "60+"]

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

    print(f"\n[Frame {frame_count}] Person Detections:")
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
                resized_crop = cv2.resize(crop, (80, 160))
                blob = np.expand_dims(np.transpose(resized_crop, (2, 0, 1)), axis=0).astype(np.float32)
                result = compiled_person_model([blob])[person_output].flatten()

                gender = "Male" if result[0] > 0.5 else "Female"
                age_idx = int(np.argmax(result[2:6]))
                age = age_groups[age_idx]
                print(f" Gender: {gender} | Age Group: {age}")
            except Exception as e:
                print(" Error in person classification:", e)

cap.release()
print("\n Person classification complete.")
