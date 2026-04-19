import cv2
import mediapipe as mp
import os
#هذا الكود ياخذ من الفيديو صورة الوجة فقط
# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained face validation model (Haar Cascade for face validation)
face_validator = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define input and output paths
input_dir = "Path to your videos directory"
output_dir = "Path to your File directory"
os.makedirs(output_dir, exist_ok=True)

# Get a list of all video files in the input directory
video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

print(f"Found {len(video_files)} video(s) to process.")

# Process each video
face_count = 0  # Global face counter for unique naming
for video_file in video_files:
    video_path = os.path.join(input_dir, video_file)
    print(f"Processing video: {video_file}")

    # Load the video
    video_capture = cv2.VideoCapture(video_path)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        frame_count = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print(f"Finished processing {video_file}.")
                break

            frame_count += 1

            # Process every Nth frame (e.g., every 10th frame)
            if frame_count % 10 != 0:
                continue

            # Convert the frame to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Extract bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(
                        bboxC.height * ih)

                    # Ensure bounding box is within frame bounds
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)

                    # Crop the face
                    if w > 0 and h > 0:  # Ensure valid dimensions
                        face = frame[y:y + h, x:x + w]

                        if face.size != 0:  # Ensure face is not empty
                            try:
                                # Convert to grayscale for Haar Cascade validation
                                gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                                detected_faces = face_validator.detectMultiScale(
                                    gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                                )

                                # Only save the face if it is validated by Haar Cascade
                                if len(detected_faces) > 0:
                                    # Resize face to 512x512
                                    face_resized = cv2.resize(face, (224, 224))

                                    # Save the resized face
                                    face_filename = os.path.join(output_dir, f"face_{face_count}.jpg")
                                    cv2.imwrite(face_filename, face_resized)
                                    face_count += 1

                            except Exception as e:
                                print(f"Error processing face crop in {video_file}: {e}")

        # Release resources for this video
        video_capture.release()

print(f"Face extraction completed. {face_count} faces saved in '{output_dir}'.")
