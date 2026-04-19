import os
import cv2
import numpy as np
from mtcnn import MTCNN  # Install via `pip install mtcnn`
#هذا الكود ياتي بعد تحسن جودة الصوره يتاكد ان الصورة وجهه و يعيد مقاسات الصوره
# Paths
real_images_path = "/Users/.../Desktop/.../real"
fake_images_path = "/Users/.../Desktop/.../fake"
output_real_path = "/Users/.../Desktop/.../real"
output_fake_path = "/Users/.../Desktop/.../fake"

# Create output directories
os.makedirs(output_real_path, exist_ok=True)
os.makedirs(output_fake_path, exist_ok=True)

# Initialize face detector
detector = MTCNN()

# Quality thresholds
MIN_RESOLUTION = (256, 256)  # Minimum resolution (width, height)
MIN_BRIGHTNESS = 30         # Minimum brightness level
MAX_BRIGHTNESS = 230         # Maximum brightness level
MIN_SHARPNESS = 50          # Minimum sharpness (Laplacian variance threshold)

#Check image quality based on resolution, brightness, and sharpness.
def check_image_quality(img):

    # Check resolution
    if img.shape[1] < MIN_RESOLUTION[0] or img.shape[0] < MIN_RESOLUTION[1]:
        return False, "Low resolution"

    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        return False, "Brightness issue"

    # Check sharpness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_SHARPNESS:
        return False, "Blurry image"

    return True, None

def clean_and_resize(input_path, output_path):
    images = os.listdir(input_path)
    processed_count = 0

    for image_name in images:
        img_path = os.path.join(input_path, image_name)
        try:
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_path}, skipping.")
                continue

            # Detect faces
            faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if len(faces) == 0:
                print(f"No face detected in {img_path}, skipping.")
                continue

            # Check quality
            quality, issue = check_image_quality(img)
            if not quality:
                print(f"Skipping {img_path} due to: {issue}")
                continue

            # Resize the image to 224x224
            img_resized = cv2.resize(img, (224, 224))

            # Save the cleaned image
            output_file = os.path.join(output_path, image_name)
            cv2.imwrite(output_file, img_resized)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Processed {processed_count} images in {input_path}.")

# Clean and resize for real and fake images
clean_and_resize(real_images_path, output_real_path)
clean_and_resize(fake_images_path, output_fake_path)
