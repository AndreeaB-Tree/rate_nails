import json
import math
import tkinter as tk
from tkinter import Label, Tk, filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="rgJkZrGlxEhM7WTHPc6H"
)

output_image_path = "output_image.jpg"
predictions_file = "predictions.json"
painted_nails_image_path = "src\\bratzslay.jpg"

def scale_image(image, max_width = 800, max_height = 800):
    """Scales image to fit within the specified dimensions."""
    height, width = image.shape[:2]
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_image


def detect_painted_nails(image, predictions):
    """Detects painted nails, while ignoring natural-colored nails."""
    painted_nail_count = 0

    for pred in predictions:
        x1 = int(round(pred['x'] - pred['width'] / 2))
        y1 = int(round(pred['y'] - pred['height'] / 2))
        x2 = int(round(pred['x'] + pred['width'] / 2))
        y2 = int(round(pred['y'] + pred['height'] / 2))

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Calculate RGB averages and brightness
        b_mean, g_mean, r_mean = cv2.mean(cropped)[:3]
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        mean_brightness = cv2.mean(gray_cropped)[0]

        is_painted = False

        # Check for white color
        if abs(r_mean - g_mean) < 15 and abs(g_mean - b_mean) < 15 and r_mean >= 170:
            is_painted = True  # White or near-white nails
        if mean_brightness >= 170:
            is_painted = True

        # RGB ranges for natural nails 
        natural_min = (180, 150, 120)
        natural_max = (255, 210, 180)
        if not (natural_min[0] <= r_mean <= natural_max[0] and
                natural_min[1] <= g_mean <= natural_max[1] and
                natural_min[2] <= b_mean <= natural_max[2]):
            if not is_painted:
                if r_mean > 150 or g_mean > 150 or b_mean > 150:
                    is_painted = True
                elif (r_mean > 100 and g_mean > 100 and b_mean > 100) and \
                     (max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean) < 50):
                    is_painted = True

        if is_painted:
            painted_nail_count += 1
            
    return painted_nail_count == len(predictions)


def preprocess_image(image):
    """Normalizes lighting using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    normalized_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return normalized_image

def apply_gaussian_blur(image):
    """Applies Gaussian blur."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def detect_cracks_and_splits(image, pred):
    """Detects cracks and splits on the nail."""
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Exclude very bright regions
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(bright_mask))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_like_contours = [
        c for c in contours 
        if cv2.arcLength(c, True) / (cv2.contourArea(c) + 1) > 3.0 and cv2.contourArea(c) > 5
    ]
    crack_count = len(crack_like_contours)

    # Return a score based on detected cracks
    return int(min(100, crack_count * 5))  # Scale -> [0, 100]


def detect_ridges(image, pred):
    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 0
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    ridge_score = np.std(laplacian)  # Higher variance indicates more pronounced ridges

    return int(min(100, ridge_score * 5))  # Scale -> [0, 100]


def measure_health(image, pred):
    """Measures nail health."""
    preprocessed_image = preprocess_image(image)
    blurred_image = apply_gaussian_blur(preprocessed_image)

    x1 = int(pred['x'] - pred['width'] / 2)
    y1 = int(pred['y'] - pred['height'] / 2)
    x2 = int(pred['x'] + pred['width'] / 2)
    y2 = int(pred['y'] + pred['height'] / 2)

    cropped = blurred_image[y1:y2, x1:x2]
    if cropped.size == 0:
        return 1

    # LAB color analysis -> for lighting  variations
    lab_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l_mean, a_mean, b_mean = cv2.mean(lab_cropped)[:3]
    is_yellow = b_mean > 150 and a_mean < 130
    is_green = a_mean < 110 and b_mean > 120

    # Laplacian for texture analysis
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # cv2.imshow("laplacian", laplacian)
    # cv2.waitKey(0)
    laplacian_var = laplacian.var() # Low variance -> smooth surface

    crack_score = detect_cracks_and_splits(blurred_image, pred)
    ridge_score = detect_ridges(blurred_image, pred)
    health_score = 5
    print(f"is yellow {is_yellow} and is green {is_green}")
    print(f"laplacian {laplacian_var}")
    print(f"crack_score {crack_score}")
    print(f"ridge_score {ridge_score}")
    print("NEXT")
    if is_yellow or is_green:
        health_score == 1

    if laplacian_var < 10:
        health_score -= 2
    elif laplacian_var < 20:
        health_score -= 1

    if crack_score > 60:
        health_score -= 2
    if crack_score > 20:
        health_score -= 1
    if ridge_score > 20:
        health_score -= 1

    return int(max(1, min(5, health_score)))


def calculate_rating(pred, image):
    ratings = {}

    aspect_ratio = pred['height'] / pred['width'] if pred['width'] > 0 else 0
    if aspect_ratio > 2.1:
        length_score = 5
    elif aspect_ratio > 1.6:
        length_score = 4
    elif aspect_ratio > 1.1:
        length_score = 3
    elif aspect_ratio > 0.8:
        length_score = 2
    else:
        length_score = 1
    ratings['length'] = length_score

    # shape = crookedness + symmetry
    crookedness = measure_crookedness(pred['points'])
    symmetry_score = measure_symmetry(pred['points'])
    shape_score = max(5 - int(crookedness / 3), symmetry_score)
    ratings['shape'] = shape_score

    # health  = smoothness + uniformity
    health_score = measure_health(image, pred)
    ratings['health'] = health_score

    return ratings


def measure_symmetry(points):
    """Evaluates symmetry by comparing left and right halves of the nail."""
    x_coords = np.array([p['x'] for p in points])
    # y_coords = np.array([p['y'] for p in points])
    center_x = np.mean(x_coords)

    left_half = np.array([p for p in points if p['x'] < center_x])
    right_half = np.array([p for p in points if p['x'] >= center_x])

    if len(left_half) == 0 or len(right_half) == 0:
        return 1 

    left_y_mean = np.mean([p['y'] for p in left_half])
    right_y_mean = np.mean([p['y'] for p in right_half])
    left_diff = np.abs(left_y_mean - right_y_mean)
    return max(1, 5 - int(left_diff / 2))


def measure_crookedness(points):
    """Measures how straight the nail is based on the points."""
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

        rating_text = [
            f"Length: {ratings['length']}",
            f"Shape: {ratings['shape']}",
            f"Health: {ratings['health']}",
        ]

        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        line_height = 30
        overlay = image.copy()
        text_box_height = line_height * len(rating_text) + 10
        text_box_width = 200

        cv2.rectangle(
            overlay,
            (x, y),
            (x + text_box_width, y + text_box_height),
            (249, 201, 225),
            -1,
        )
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for i, line in enumerate(rating_text):
            cv2.putText(
                image,
                line,
                (x + 10, y + 20 + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (233, 55, 144),
                2,
                cv2.LINE_AA,
            )
    return image

def display_low_health_warning():
    """Displays a window to indicate that a nail has a health rating of 1 or 2."""
    low_health_window = tk.Toplevel()
    low_health_window.title("Low Health Warning")
    low_health_window.geometry("600x650")
    low_health_window.configure(bg="#FFC0CB")

    message_label = tk.Label(
        low_health_window,
        text="⚠️ Low Health Detected!",
        font=("Comic Sans MS", 16, "bold"),
        bg="#FFC0CB",
        fg="#C71585",
    )
    message_label.pack(pady=10)
    try:
        image_path = "src\\shockedbratz.jpg"
        low_health_image = Image.open(image_path)
        low_health_image = low_health_image.resize((400, 400), Image.Resampling.LANCZOS)
        low_health_image_tk = ImageTk.PhotoImage(low_health_image)

        image_label = tk.Label(low_health_window, image=low_health_image_tk, bg="#FFC0CB")
        image_label.image = low_health_image_tk
        image_label.pack(pady=10)
    except Exception as e:
        print("Error loading image for low health warning:", e)

    explanation_label = tk.Label(
        low_health_window,
        text=("Please check your nails' condition and consider taking care of them!"),
        font=("Comic Sans MS", 12),
        bg="#FFC0CB",
        fg="#800080",
        justify="center",
        wraplength=350,
    )
    explanation_label.pack(pady=10)

    close_button = ttk.Button(
        low_health_window,
        text="Close",
        command=low_health_window.destroy,
    )
    close_button.pack(pady=20)


def display_no_nails_detected():
    """Displays a window to indicate no nails were detected."""
    no_nails_window = tk.Toplevel()
    no_nails_window.title("No Nails Detected")
    no_nails_window.geometry("600x650")
    no_nails_window.configure(bg="#FFC0CB")

    message_label = tk.Label(
        no_nails_window,
        text="😔 No nails detected in the image!",
        font=("Comic Sans MS", 16, "bold"),
        bg="#FFC0CB",
        fg="#C71585",
    )
    message_label.pack(pady=10)
    try:
        image_path = "src\\sadbratz.jpg"
        no_nails_image = Image.open(image_path)
        no_nails_image = no_nails_image.resize((400, 400), Image.Resampling.LANCZOS)
        no_nails_image_tk = ImageTk.PhotoImage(no_nails_image)

        image_label = tk.Label(no_nails_window, image=no_nails_image_tk, bg="#FFC0CB")
        image_label.image = no_nails_image_tk
        image_label.pack(pady=10)
    except Exception as e:
        print("Error loading image for no nails detected:", e)

    explanation_label = tk.Label(
        no_nails_window,
        text=(
            "Please ensure the image clearly shows nails.\n"
            "Try adjusting lighting or angle and try again."
        ),
        font=("Comic Sans MS", 12),
        bg="#FFC0CB",
        fg="#800080",
        justify="center",
        wraplength=350,
    )
    explanation_label.pack(pady=10)
    close_button = ttk.Button(
        no_nails_window,
        text="Close",
        command=no_nails_window.destroy,
    )
    close_button.pack(pady=20)



def display_results_with_guide(annotated_image_path):
    """Displays the annotated image and rating guide in a new tkinter window."""
    annotated_image = Image.open(annotated_image_path)
    annotated_image_np = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    scaled_annotated_image = scale_image(annotated_image_np, 650, 650)
    scaled_annotated_image_pil = Image.fromarray(cv2.cvtColor(scaled_annotated_image, cv2.COLOR_BGR2RGB))
    scaled_annotated_image_tk = ImageTk.PhotoImage(scaled_annotated_image_pil)

    result_window = tk.Toplevel()
    result_window.title("Nail Detection Results")
    result_window.geometry("1100x750")
    result_window.configure(bg="#FFC0CB")

    image_label = Label(result_window, image=scaled_annotated_image_tk, bg="#FFC0CB")
    image_label.image = scaled_annotated_image_tk
    image_label.pack(side="left", padx=20, pady=20)

    guide_text = (
        "✨ **Length**:\n"
        "1 - Not a good length\n"
        "3 - Average\n"
        "5 - Good proportional length\n\n"
        "✨ **Shape**:\n"
        "1 - Crooked or asymmetrical\n"
        "3 - Slightly uneven\n"
        "5 - Symmetrical and aligned\n\n"
        "✨ **Health**:\n"
        "1 - Cracked, ridged, or discolored\n"
        "3 - Minor issues (e.g., faint ridges)\n"
        "5 - Smooth and uniform"
    )

    guide_label = Label(
        result_window,
        text="Nail Rating Guide",
        font=("Comic Sans MS", 16, "bold"),
        bg="#FFC0CB",
        fg="#C71585",
    )
    guide_label.pack(pady=10, side="top")

    guide_text_label = Label(
        result_window,
        text=guide_text,
        font=("Comic Sans MS", 14),
        bg="#FFC0CB",
        fg="#800080",
        justify="left",
        wraplength=300,
    )
    guide_text_label.pack(side="right", padx=20, pady=20)

    ttk.Button(
        result_window,
        text="Close",
        command=result_window.destroy,
    ).pack(pady=10, side="bottom")


def process_image(image_path):
    """Processes the image and calls display_results_with_guide to show the results."""
    try:
        # Perform inference
        result = CLIENT.infer(image_path, model_id="nails_segmentation-vhnmw-p6sip/3")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open the image at '{image_path}'.")
            return

        if not result['predictions']:
            display_no_nails_detected()
            return

        all_ratings = []
        for pred in result['predictions']:
            ratings = calculate_rating(pred, image)
            all_ratings.append({"prediction": pred, "ratings": ratings})

        annotated_image = image.copy()
        for item in all_ratings:
            pred = item["prediction"]
            ratings = item["ratings"]
            annotated_image = annotate_image_with_ratings(annotated_image, [pred])

        cv2.imwrite(output_image_path, annotated_image)

        low_health_detected = any(item["ratings"]["health"] in [1, 2] for item in all_ratings)

        if low_health_detected:
            display_low_health_warning()
        display_results_with_guide(output_image_path)

    except Exception as e:
        print("Error during inference:", e)

# FOR LIVE RATINGS
# def process_frame(frame):
#     try:
#         temp_image_path = "temp_frame.jpg"
#         cv2.imwrite(temp_image_path, frame)
#         result = CLIENT.infer(temp_image_path, model_id="nails_segmentation-vhnmw-p6sip/3")

#         annotated_frame = annotate_image_with_ratings(frame, result['predictions'])
#         return annotated_frame

#     except Exception as e:
#         print("Error during inference:", e)
#         return frame

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
            cv2.imshow("Webcam Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # When 's' is pressed, take a snapshot
                print("Snapshot taken. Processing...")
                temp_image_path = "snapshot.jpg"
                cv2.imwrite(temp_image_path, frame)
                cap.release()
                cv2.destroyAllWindows()
                
                # Perform inference
                try:
                    result = CLIENT.infer(temp_image_path, model_id="nails_segmentation-vhnmw-p6sip/3")
                    image = cv2.imread(temp_image_path)
                    if image is None:
                        print(f"Error: Could not open the snapshot at '{temp_image_path}'.")
                        return

                    if not result['predictions']:
                        display_no_nails_detected()
                        return

                    all_ratings = []
                    for pred in result['predictions']:
                        ratings = calculate_rating(pred, image)
                        all_ratings.append({"prediction": pred, "ratings": ratings})

                    annotated_image = image.copy()
                    for item in all_ratings:
                        pred = item["prediction"]
                        ratings = item["ratings"]
                        annotated_image = annotate_image_with_ratings(annotated_image, [pred])
                    annotated_image_path = "annotated_snapshot.jpg"
                    cv2.imwrite(annotated_image_path, annotated_image)

                    low_health_detected = any(item["ratings"]["health"] in [1, 2] for item in all_ratings)

                    if low_health_detected:
                        display_low_health_warning()
                    display_results_with_guide(annotated_image_path)

                except Exception as e:
                    print("Error during inference:", e)

                break

            elif key == ord('q'):
                print("Exiting webcam...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    def choose_photo():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.png;*.jpeg")])
        if file_path:
            process_image(file_path)

    def use_webcam():
        start_webcam()

    def update_gif(frame_index=0):
        try:
            frame = gif_frames[frame_index]
            gif_label.config(image=frame)
            root.after(gif_delays[frame_index], update_gif, (frame_index + 1) % len(gif_frames))
        except Exception as e:
            print("Error updating GIF:", e)

    root = Tk()
    root.title("Nail Detection")
    root.geometry("400x400")
    root.configure(bg="#FFC0CB")
    title_font = ("Comic Sans MS", 18, "bold")
    button_font = ("Comic Sans MS", 12)
    Label(root, text="✨💅 Nail Detection 💅✨", font=title_font, bg="#FFC0CB", fg="#C71585").pack(pady=10)

    style = ttk.Style()
    style.configure(
        "TButton",
        font=button_font,
        foreground="#C71585",
        background="#FF69B4",
        borderwidth=1,
    )
    style.map(
        "TButton",
        background=[("active", "#FF1493")],
        foreground=[("active", "#C71585")],
    )

    ttk.Button(root, text="Upload Photo", command=choose_photo).pack(pady=10)
    ttk.Button(root, text="Use Webcam", command=use_webcam).pack(pady=10)

    try:
        gif_path = "src\\hotbratz.gif"
        gif_image = Image.open(gif_path)
        gif_frames = []
        gif_delays = []

        while True:
            frame = ImageTk.PhotoImage(gif_image.copy())
            gif_frames.append(frame)
            gif_delays.append(gif_image.info.get("duration", 100))
            gif_image.seek(gif_image.tell() + 1)
    except EOFError:
        pass
    except Exception as e:
        print("Error loading GIF:", e)

    gif_label = Label(root, bg="#FFC0CB")
    gif_label.pack(pady=10)
    if gif_frames:
        update_gif()

    root.mainloop()


if __name__ == "__main__":
    main()
