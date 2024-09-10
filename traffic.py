
import cv2
import numpy as np
import os
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

# Function to get YOLO network
def get_yolo_net(cfg_file, weights_file, names_file):
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    if not os.path.exists(names_file):
        raise FileNotFoundError(f"Names file not found: {names_file}")

    net = cv2.dnn.readNet(weights_file, cfg_file)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

# Function to detect objects using YOLO
def detect_objects(net, output_layers, frame, input_size=(416, 416)):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [0, 1, 2]:  # Only detect persons, bicycles, and cars
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        final_indices = indices.flatten().tolist()
        final_boxes = [boxes[i] for i in final_indices]
        final_confidences = [confidences[i] for i in final_indices]
        final_class_ids = [class_ids[i] for i in final_indices]
    else:
        final_boxes = []
        final_confidences = []
        final_class_ids = []

    return final_boxes, final_confidences, final_class_ids

# Function to draw labels on the frame
def draw_labels(frame, trackers, fps, pixel_per_meter):
    for tracker in trackers:
        if not tracker.is_confirmed() or tracker.time_since_update > 1:
            continue

        bbox = tracker.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if hasattr(tracker, 'previous_center'):
            previous_center = tracker.previous_center
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            speed = calculate_speed(previous_center, current_center, fps, pixel_per_meter)
            tracker.speed = speed
            tracker.previous_center = current_center
        else:
            tracker.previous_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            tracker.speed = 0

        label = tracker.get_label()
        cv2.putText(frame, f"{label} {tracker.speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to calculate speed
def calculate_speed(previous_center, current_center, fps, pixel_per_meter):
    distance_pixels = np.linalg.norm(np.array(current_center) - np.array(previous_center))
    distance_meters = distance_pixels * pixel_per_meter
    speed_mps = distance_meters * fps
    speed_kmph = speed_mps * 3.6
    return speed_kmph

# Main function
def main(video_path):
    cfg_file = "yolov4-tiny.cfg"
    weights_file = "yolov4-tiny.weights"
    names_file = "coco.names"

    try:
        net, output_layers, classes = get_yolo_net(cfg_file, weights_file, names_file)
    except FileNotFoundError as e:
        print(e)
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps // 10))  # Run detection every half a second

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    deepsort = DeepSort(max_age=30, n_init=3)
    pixel_per_meter = 0.5  # Each pixel corresponds to 2 meters in the real world
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Process every frame for tracking
        if frame_count % frame_interval == 0:
            # Process at a lower resolution for faster detection
            input_frame = cv2.resize(frame, (416, 416))
            boxes, confidences, class_ids = detect_objects(net, output_layers, input_frame)
            detections = []
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                # Scale the bounding box back to the original frame size
                scale_x = frame.shape[1] / 416
                scale_y = frame.shape[0] / 416
                box = (int(box[0] * scale_x), int(box[1] * scale_y), int(box[2] * scale_x), int(box[3] * scale_y))
                detections.append((box, conf, classes[class_id]))
            trackers = deepsort.update_tracks(detections, frame)
        else:
            trackers = deepsort.update_tracks([], frame)

        # Draw bounding boxes for all detections
        for (box, conf, label) in detections:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        draw_labels(frame, trackers, fps, pixel_per_meter)

        # Resize the frame to 1/3rd of the screen width
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        scale = 1/3
        resized_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))

        cv2.imshow('Car Detection & Tracking', resized_frame)

        elapsed_time = time.time() - start_time
        wait_time = max(1, int((1000 / fps) - (elapsed_time * 1000)))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    main(video_path)
