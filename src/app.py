from inference_sdk import InferenceHTTPClient
import cv2
import json
import tkinter as tk
from tkinter import filedialog

import math
import numpy as np

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="rgJkZrGlxEhM7WTHPc6H"
)

# Output paths
output_image_path = "output_image.jpg"
predictions_file = "predictions.json"
painted_nails_image_path = "src\\bratzslay.jpg"

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


def detect_painted_nails(image, predictions):
    """
    Detect painted nails, including white-painted nails, while ignoring natural-colored nails.
    """
    painted_nail_count = 0

    for pred in predictions:
        x1 = int(round(pred['x'] - pred['width'] / 2))
        y1 = int(round(pred['y'] - pred['height'] / 2))
        x2 = int(round(pred['x'] + pred['width'] / 2))
        y2 = int(round(pred['y'] + pred['height'] / 2))

        # Ensure indices are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Calculate RGB channel means and brightness
        b_mean, g_mean, r_mean = cv2.mean(cropped)[:3]
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray_cropped)[0]

        # Detect painted nails
        is_painted = False

        # Check for white or near-white color (r \u2248 g \u2248 b) with high brightness
        if abs(r_mean - g_mean) < 15 and abs(g_mean - b_mean) < 15 and r_mean >= 170:
            is_painted = True  # White or near-white nails

        # Additional check for high brightness to confirm white-painted nails
        if mean_brightness >= 170:
            is_painted = True

        # Define RGB ranges for natural nails (skin tones)
        natural_min = (180, 150, 120)  # Lower bounds for natural colors (R, G, B)
        natural_max = (255, 210, 180)  # Upper bounds for natural colors (R, G, B)

        # Check if nail color is not within the natural nail color range
        if not (natural_min[0] <= r_mean <= natural_max[0] and
                natural_min[1] <= g_mean <= natural_max[1] and
                natural_min[2] <= b_mean <= natural_max[2]):
            # Detect painted nails in bright or pastel colors if not already classified as white
            if not is_painted:  # Check for bright/pastel colors if not already classified as white
                if r_mean > 150 or g_mean > 150 or b_mean > 150:  # Bright colors
                    is_painted = True
                elif (r_mean > 100 and g_mean > 100 and b_mean > 100) and \
                     (max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean) < 50):  # Pastels
                    is_painted = True

        if is_painted:
            painted_nail_count += 1

    return painted_nail_count == len(predictions)

def measure_smoothness(image, pred):
    """Measure health based on texture and color analysis."""
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 99999  # Large penalty for empty crop

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Texture analysis: Laplacian variance (higher variance => more "roughness")
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color analysis: Check for discoloration using HSV
    hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h_std, s_std, v_std = np.std(hsv_cropped, axis=(0, 1))

    texture_score = max(0, 100 - laplacian_var)  
    color_score = max(0, 100 - (h_std + s_std + v_std) / 3)

    health_score = (texture_score + color_score) / 2
    return max(0, min(100, health_score))  # clamp to [0,100]

def preprocess_image(image):
    """Normalize lighting using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge the channels back
    lab_clahe = cv2.merge((l_clahe, a, b))
    normalized_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return normalized_image

def apply_gaussian_blur(image):
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def detect_cracks_and_splits(image, pred):
    """Detect cracks and splits on the nail surface."""
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0  # No cracks detected

    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Exclude very bright regions (reflections)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(bright_mask))

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by shape (e.g., length-to-width ratio)
    crack_like_contours = [
        c for c in contours if cv2.arcLength(c, True) / (cv2.contourArea(c) + 1) > 2.5
    ]

    # Count crack-like contours
    crack_count = len(crack_like_contours)

    # Return a crack score based on detected cracks
    return int(min(100, crack_count * 10))  # Scale score to [0, 100]


def detect_ridges(image, pred):
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0  # No ridges detected

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)

    # Apply Laplacian for ridge detection
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    ridge_score = np.std(laplacian)  # Higher variance indicates more pronounced ridges

    # Normalize and return score
    return int(min(100, ridge_score * 10))


def measure_health(image, pred):
    """Measure nail health with lighting normalization and enhanced analysis."""
    # Preprocess the image for better analysis
    preprocessed_image = preprocess_image(image)
    blurred_image = apply_gaussian_blur(preprocessed_image)

    # Proceed with existing LAB color analysis and Laplacian calculations
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = blurred_image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 1  # Assign minimum health score for invalid regions

    # LAB color analysis
    lab_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_mean, a_mean, b_mean = cv2.mean(lab_cropped)[:3]
    is_yellow = b_mean > 150 and a_mean < 130
    is_green = a_mean < 110 and b_mean > 120

    # Laplacian for texture analysis
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # Detect cracks and splits
    crack_score = detect_cracks_and_splits(blurred_image, pred)

    # Detect ridges
    ridge_score = detect_ridges(blurred_image, pred)

    # Combine metrics for health score
    health_score = 5  # Start with a perfect health score

    # Penalize for discoloration
    if is_yellow or is_green:
        health_score -= 1

    # Penalize for texture roughness
    if laplacian_var < 10:
        health_score -= 2
    elif laplacian_var < 20:
        health_score -= 1

    # Penalize for cracks and ridges
    if crack_score > 20:  # Adjust threshold as needed
        health_score -= 1
    if ridge_score > 20:  # Adjust threshold as needed
        health_score -= 1

    # Clamp the health score to a range [1, 5]
    return int(max(1, min(5, health_score)))






def calculate_rating(pred, image):
    ratings = {}

    # LENGTH
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

    # SHAPE (crookedness + symmetry)
    crookedness = measure_crookedness(pred['points'])
    symmetry_score = measure_symmetry(pred['points'])
    shape_score = max(5 - int(crookedness / 3), symmetry_score)
    ratings['shape'] = shape_score

    # HEALTH (SMOOTHNESS AND UNIFORMITY)
    health_score = measure_health(image, pred)
    ratings['health'] = health_score

    return ratings



def measure_symmetry(points):
    """Evaluate symmetry by comparing left and right halves of the nail."""
    x_coords = np.array([p['x'] for p in points])
    y_coords = np.array([p['y'] for p in points])
    center_x = np.mean(x_coords)

    left_half = np.array([p for p in points if p['x'] < center_x])
    right_half = np.array([p for p in points if p['x'] >= center_x])

    if len(left_half) == 0 or len(right_half) == 0:
        return 1  # Penalize missing halves

    left_y_mean = np.mean([p['y'] for p in left_half])
    right_y_mean = np.mean([p['y'] for p in right_half])
    left_diff = np.abs(left_y_mean - right_y_mean)
    return max(1, 5 - int(left_diff / 2))


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
        rating_text = [f"Length: {ratings['length']} ",
                       f"Shape: {ratings['shape']} ",
                       f"Health: {ratings['health']}"]

        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        line_height = 30
        # Render each line
        for i, line in enumerate(rating_text):
            cv2.putText(
                image,
                line,
                (x, y + i * line_height),  # Adjust y-position for each line
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (214, 81, 188),
                2,
                cv2.LINE_AA
            )
    return image


def process_image(image_path):
    try:
        # Perform inference
        result = CLIENT.infer(image_path, model_id="nails_segmentation-vhnmw-p6sip/3")

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open the image at '{image_path}'.")
            return

        # Check for painted nails
        if detect_painted_nails(image, result['predictions']):
            print("Painted nails detected!")
            
            # Attempt to load the "painted nails detected" image
            painted_image = cv2.imread(painted_nails_image_path)
            if painted_image is not None:
                cv2.imshow("Painted Nails Detected", scale_image(painted_image))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                # If the file can't be read, avoid .shape error
                print(f"Warning: Could not open '{painted_nails_image_path}'. Skipping that display.")
            return

        # Annotate the image with ratings
        annotated_image = annotate_image_with_ratings(image, result['predictions'])

        # Save and display the processed image
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Output image saved to {output_image_path}")

        # Save predictions to a JSON file
        with open(predictions_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Predictions saved to {predictions_file}")

        # Display the annotated image
        cv2.imshow("Processed Image", scale_image(annotated_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error during inference:", e)

def process_frame(frame):
    try:
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

            # Display the live webcam feed
            cv2.imshow("Webcam Feed", frame)

            # Check for user input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Take a snapshot and process it
                print("Snapshot taken. Processing...")

                # Save the current frame to a temporary file
                temp_image_path = "snapshot.jpg"
                cv2.imwrite(temp_image_path, frame)

                # Process the snapshot
                process_image(temp_image_path)

            elif key == ord('q'):  # Exit the webcam feed
                print("Exiting webcam...")
                break

    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

# Ensure the main function remains unchanged
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
