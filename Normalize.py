import cv2
import os
import numpy as np

# هذا الكود يحسن جودة الصورة
input_folder = "/Users/.../Desktop/.../fake"
output_folder = "/Users/.../Desktop/.../fake"
os.makedirs(output_folder, exist_ok=True)

# Adjust gamma correction for brightness normalization.
def adjust_gamma(image, gamma=1.2):

    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# sharpening filter to enhance image clarity.
def sharpen_image(image):

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# Process all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Normalize brightness
        image = adjust_gamma(image, gamma=1.2)

        # Normalize sharpness
        image = sharpen_image(image)

        # Save processed image
        cv2.imwrite(os.path.join(output_folder, filename), image)

print("Image normalization completed! Processed images saved in:", output_folder)
