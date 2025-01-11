import cv2
from ultralytics import YOLO

def rate_nails():
    # Load the trained YOLO segmentation model
    model = YOLO("../runs/segment/train8/weights/best.pt")

    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Run the model on the current frame
        results = model(frame)

        for result in results:
            # Check if detections exist
            if result.boxes is None or result.masks is None:
                continue  # Skip this frame if no detections

            # Process detected nails
            for box, single_mask in zip(result.boxes, result.masks):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                nail_length = y2 - y1  # Approximate length
                rating = min(10, nail_length / 30 * 10)  # Scale to max 3cm
                
                # Draw bounding boxes and rating
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Rate: {rating:.1f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the frame
        cv2.imshow("Nail Rating App", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rate_nails()
