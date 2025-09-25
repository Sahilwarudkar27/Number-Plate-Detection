import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ------------------------------
# Streamlit App Title
# ------------------------------
st.title("Number Plate Detection using Yolo 🚗")


# ------------------------------
# Load YOLO Model
# ------------------------------
try:
    model = YOLO(r"C:\Users\sahil\OneDrive\Desktop\Plate\best_license_plate_model.pt")
    st.success("✅ YOLO model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading YOLO model: {e}")

# ------------------------------
# Prediction Functions
# ------------------------------
def predict_and_save_image(path_test_car: str, output_image_path: str) -> str:
    """
    Predict bounding boxes on an image using YOLO and save the output image.
    
    Args:
        path_test_car (str): Path to the input image.
        output_image_path (str): Path to save the output image with bounding boxes.

    Returns:
        str: Path to the saved output image.
    """
    try:
        # Run YOLO prediction
        results = model.predict(path_test_car, device="cpu")
        
        # Read image using OpenCV
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes and confidence on detected objects
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{confidence*100:.2f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert back to BGR for saving with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)

        return output_image_path

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None


def predict_and_plot_video(video_path: str, output_path: str) -> str:
    """
    Predict bounding boxes on a video using YOLO and save the output video.
    
    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the output video with bounding boxes.

    Returns:
        str: Path to the saved output video.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ Error opening video file: {video_path}")
            return None

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device="cpu")

            # Draw bounding boxes and confidence
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence*100:.2f}%", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        return output_path

    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
        return None


def process_media(input_path: str, output_path: str) -> str:
    """
    Determines the type of uploaded file and processes it accordingly (image or video).
    
    Args:
        input_path (str): Path to the uploaded file.
        output_path (str): Path to save the processed output.

    Returns:
        str: Path to the processed output file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"❌ Unsupported file type: {file_extension}")
        return None


# ------------------------------
# File Uploader
# ------------------------------
uploaded_file = st.file_uploader("📂 Upload an image or video", 
                                 type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Ensure required folders exist
    os.makedirs("temp", exist_ok=True)  
    os.makedirs("output", exist_ok=True)  

    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("output", f"output_{uploaded_file.name}")  # Save to output folder

    try:
        # Save uploaded file to temp folder
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Show processing spinner
        with st.spinner("⚙️ Processing... Please wait..."):
            result_path = process_media(input_path, output_path)

        # Display processed result
        if result_path:
            st.success(f"✅ Output ")
            if input_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
                st.video(result_path)
            else:
                st.image(result_path)

    except Exception as e:
        st.error(f"❌ Error uploading or processing file: {e}")
