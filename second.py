import cv2
from ultralytics import YOLO

def main():
    model_name = "model.pt"
    print(f"Loading `{model_name}` model...")
    # model = YOLO("yolo11n.pt")
    model = YOLO("model.pt")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("INFO: Starting webcam. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        # stream=True is more memory efficient for video loops
        # verbose=False keeps the terminal output clean
        results = model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = f"Person: {confidence:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # for box in boxes:
            #     # Get the class ID (0 is 'person' in standard COCO models)
            #     class_id = int(box.cls[0])

            #     # Filter: Only draw box if detected class is a Person (ID 0)
            #     # If you use a custom face-trained model, you might not need this check.
            #     if class_id == 0:
            #         # Get coordinates (x1, y1, x2, y2)
            #         x1, y1, x2, y2 = map(int, box.xyxy[0])

            #         # Get confidence score
            #         confidence = float(box.conf[0])

            #         # Draw the bounding box (Color: Green, Thickness: 2)
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #         # Add label text "Person" or "Face" depending on your model
            #         label = f"Person: {confidence:.2f}"

            #         # Calculate text position (just above the box)
            #         t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
            #         c2 = x1 + t_size[0], y1 - t_size[1] - 3

            #         # Draw filled rectangle for text background for better visibility
            #         cv2.rectangle(frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)

            #         # Put the text
            #         cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("Face Blur", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
