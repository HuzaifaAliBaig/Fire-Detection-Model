import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone

# Global variables for OpenCV-related objects and flags
cap = None
is_camera_on = False
frame_count = 0
frame_skip_threshold = 3
video_paused = False

# Function to read classes from a file
def read_classes_from_file(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Function to ask the user for a model
def select_model():
    print("Please select a model (e.g. 'bestfire'): ")
    model_name = input()
    return model_name

# Function to update the OpenCV window with the live video stream
def update_stream(model, class_list):
    global frame_count, video_paused
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if frame_count % frame_skip_threshold != 0:
                continue

            # Resize frame and perform object detection
            frame = cv2.resize(frame, (1020, 500))
            results = model.track(source=frame, conf=0.4, persist=True, tracker="bytetrack.yaml")

            # Process detection results
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            for index, row in px.iterrows():
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[2])
                y2 = int(row[3])
                t = int(row[4])
                if len(row) == 7:
                    p = row[5]
                    d = int(row[6])
                    c = class_list[d]
                    p = "{:.2f}".format(p)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cvzone.putTextRect(frame, f'id:{t} {c} {p}', (x1, y1 + 20), 1, 1, (255, 255, 255), (255, 0, 0))
                else:
                    p = row[4]
                    d = int(row[5])
                    c = class_list[d]
                    p = "{:.2f}".format(p)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cvzone.putTextRect(frame, f'{c} {p}', (x1, y1 + 20), 1, 1, (255, 255, 255), (255, 0, 0))

            # Display the frame using OpenCV's `imshow`
            cv2.imshow("Fire and Smoke Detection Tracking YOLOv8", frame)

            # Wait for a short period (10ms) and handle keyboard input
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Ask user for a model
    model_name = select_model()
    
    # Construct the model path and load the model
    model_path = f"./models/{model_name}.pt"
    model = YOLO(model_path)
    
    # Load class list based on the selected model
    if model_name == "bestfire":
        class_list = read_classes_from_file('fireSmoke.txt')
    else:
        class_list = read_classes_from_file('coco.txt')
    
    # Open the default webcam
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Start live video stream and apply the model
    update_stream(model, class_list)

if __name__ == "__main__":
    main()
