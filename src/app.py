from inference_sdk import InferenceHTTPClient
import cv2
import json
import tkinter as tk
from tkinter import filedialog

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="rgJkZrGlxEhM7WTHPc6H"
)

# Output paths
output_image_path = "output_image.jpg"
predictions_file = "predictions.json"

def scale_image(image, max_width = 800, max_height = 800):
    """Scale image to fit within the specified dimensions while maintaining aspect ratio."""
    height, width = image.shape[:2]

    # Calculate scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize the image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_image

def measure_smoothness(image, pred):
    """Measure health based on texture and color analysis."""
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 99999  # Large penalty for empty crop

    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Texture analysis: Laplacian variance (higher variance indicates roughness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color analysis: Check for discoloration using HSV
    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h_std, s_std, v_std = np.std(hsv_cropped, axis=(0, 1))

    # Combine metrics: Higher Laplacian variance and color deviations indicate poor health
    texture_score = max(0, 100 - laplacian_var)  # Normalize to a 0-100 scale
    color_score = max(0, 100 - (h_std + s_std + v_std) / 3)

    # Combine texture and color scores
    health_score = (texture_score + color_score) / 2
    return max(0, min(100, health_score))  # Ensure score is between 0 and 100


def calculate_rating(pred, image):
    ratings = {}

    # LENGTH: Improved aspect ratio evaluation
    aspect_ratio = pred['height'] / pred['width'] if pred['width'] > 0 else 0
    if aspect_ratio > 2.5:
        length_score = 5
    elif aspect_ratio > 1.8:
        length_score = 4
    elif aspect_ratio > 1.3:
        length_score = 3
    elif aspect_ratio > 1.0:
        length_score = 2
    else:
        length_score = 1
    ratings['length'] = length_score

    # SHAPE: Combine crookedness and symmetry
    crookedness = measure_crookedness(pred['points'])
    symmetry_score = measure_symmetry(pred['points'])  # New function for symmetry
    shape_score = max(5 - int(crookedness / 3), symmetry_score)  # Combine scores
    ratings['shape'] = shape_score

    # HEALTH (SMOOTHNESS): Refined thresholds
    std_dev = measure_smoothness(image, pred)
    if std_dev > 90:
        health_score = 5  # Excellent health
    elif std_dev > 75:
        health_score = 4  # Good health
    elif std_dev > 50:
        health_score = 3  # Average health
    elif std_dev > 25:
        health_score = 2  # Poor health
    else:
        health_score = 1  # Very poor health
    ratings['health'] = health_score


    return ratings


def measure_symmetry(points):
    """Evaluate symmetry by comparing left and right halves of the nail."""
    x_coords = np.array([p['x'] for p in points])
    y_coords = np.array([p['y'] for p in points])
    center_x = np.mean(x_coords)

    # Split into left and right halves
    left_half = np.array([p for p in points if p['x'] < center_x])
    right_half = np.array([p for p in points if p['x'] >= center_x])

    if len(left_half) == 0 or len(right_half) == 0:
        return 1  # Penalize missing halves

    # Compare mean positions of halves
    left_y_mean = np.mean([p['y'] for p in left_half])
    right_y_mean = np.mean([p['y'] for p in right_half])
    left_diff = np.abs(left_y_mean - right_y_mean)
    return max(1, 5 - int(left_diff / 2))  # Higher score for better symmetry


def measure_color_uniformity(image, pred):
    """Measure color uniformity in the nail area."""
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0

    # Convert to HSV for better color analysis
    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h_std, s_std, v_std = np.std(hsv_cropped, axis=(0, 1))
    uniformity_score = 100 - (h_std + s_std + v_std) / 3
    return max(0, min(100, uniformity_score))

import math
import numpy as np

def measure_crookedness(points):
    """Measure how straight the nail is based on the points."""
    x_coords = np.array([int(p['x']) for p in points])
    y_coords = np.array([int(p['y']) for p in points])
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    distances = [math.hypot(x - centroid_x, y - centroid_y)
                 for x, y in zip(x_coords, y_coords)]

    std_dev = np.std(distances)
    return std_dev



def annotate_image_with_ratings(image, predictions):
    for pred in predictions:
        ratings = calculate_rating(pred, image)

        # Format detailed metrics for overlay
        rating_text = (f"Length: {ratings['length']}, "
                       f"Shape: {ratings['shape']}, "
                       f"Health: {ratings['health']}")

        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)

        cv2.putText(image,
                    rating_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA)

    return image


def process_image(image_path):
    try:
        # Perform inference
        result = CLIENT.infer(image_path, model_id="nails_segmentation-vhnmw-p6sip/3")
        
        # Load the image
        image = cv2.imread(image_path)

        # Annotate the image with ratings
        annotated_image = annotate_image_with_ratings(image, result['predictions'])

        # Save and display the processed image
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Output image saved to {output_image_path}")

        # Save predictions to a JSON file
        with open(predictions_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Predictions saved to {predictions_file}")

        # Display the image
        cv2.imshow("Processed Image", scale_image(annotated_image, 800, 800))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error during inference:", e)

def process_frame(frame):
    try:
        # Save the current frame to a temporary file
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Perform inference
        result = CLIENT.infer(temp_image_path, model_id="nails_segmentation-vhnmw-p6sip/3")

        # Annotate the frame with ratings
        annotated_frame = annotate_image_with_ratings(frame, result['predictions'])

        return annotated_frame

    except Exception as e:
        print("Error during inference:", e)
        return frame

def start_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Process the frame
            processed_frame = process_frame(frame)

            # Display the frame
            cv2.imshow("Webcam Feed with Detections", processed_frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

def main():
    def choose_photo():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.png;*.jpeg")])
        if file_path:
            process_image(file_path)

    def use_webcam():
        start_webcam()

    root = tk.Tk()
    root.title("Nail Detection")
    root.geometry("300x200")
    tk.Label(root, text="Choose an input method:", font=("Arial", 16)).pack(pady=20)
    tk.Button(root, text="Upload Photo", command=choose_photo).pack(pady=10)
    tk.Button(root, text="Use Webcam", command=use_webcam).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
