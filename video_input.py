import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone

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

# Function to process the video and detect fire
def process_video(input_video_path, output_video_path, model, class_list):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to save the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    frame_skip_threshold = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip_threshold != 0:
            continue

        # Perform object detection
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

        # Write the processed frame to the output video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
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
    
    # Input and output video paths
    input_video_path = 'input_video.mp4'  # Replace with your input video path
    output_video_path = 'output_video.mp4'  # Replace with your desired output video path

    # Process the video and apply the model
    process_video(input_video_path, output_video_path, model, class_list)

if __name__ == "__main__":
    main()
