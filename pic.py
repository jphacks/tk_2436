import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_facial_landmarks(image):
    """
    Detects facial landmarks in an image using MediaPipe.
    Returns a list of (x, y) tuples representing landmark positions.
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    landmarks = []
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        landmarks.append((x, y))
    return landmarks

def create_facial_region_mask(image, landmarks, region_indices):
    """
    Creates a mask for a specific facial region based on landmark indices.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = [landmarks[i] for i in region_indices]
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def adjust_skin_color(image, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0, region_mask=None):
    """
    Adjusts the skin tone of specified facial regions in an image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if region_mask is not None:
        skin_mask = region_mask
    else:
        # Define skin color range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # Extract skin regions
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Convert skin to HSV for adjustments
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(skin_hsv)

    # Adjust Hue
    h = np.mod(h.astype(np.int32) + hue_shift, 180).astype(np.uint8)

    # Adjust Saturation
    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Adjust Lightness (Value)
    v = cv2.multiply(v, lightness_scale)
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Merge channels back
    adjusted_hsv = cv2.merge((h, s, v))
    adjusted_skin = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    # Combine adjusted skin with the original image
    inverse_skin_mask = cv2.bitwise_not(skin_mask)
    background = cv2.bitwise_and(image, image, mask=inverse_skin_mask)
    result = cv2.add(background, adjusted_skin)

    return result

def main():
    # Load the image
    image_path = 'path_to_your_image.jpg'  # Replace with your image path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Get facial landmarks
    landmarks = get_facial_landmarks(image)
    if landmarks is None:
        print("Error: No face detected.")
        return

    # Define facial regions (Added more regions relevant to cosmetics)
    regions = {
        "Left Cheek": [234, 93, 137, 177, 132, 58, 172, 136, 215, 138, 213, 147, 123],
        "Right Cheek": [454, 323, 366, 397, 362, 288, 391, 365, 416, 367, 414, 377, 152],
        "Upper Lip": list(range(61, 68)) + list(range(0, 7)),
        "Lower Lip": list(range(146, 150)) + list(range(181, 185)),
        "Left Eye": list(range(33, 42)) + list(range(133, 142)),
        "Right Eye": list(range(362, 371)) + list(range(263, 272)),
        "Left Eyebrow": list(range(55, 65)),
        "Right Eyebrow": list(range(285, 295)),
        "Nose": list(range(6, 20)) + list(range(195, 205)),
        "Forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
        "Chin": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93],
        "Jawline": list(range(5, 17)) + list(range(205, 216))
    }

    # Edits: Added new regions "Left Eyebrow", "Right Eyebrow", "Nose", "Forehead", "Chin", "Jawline"

    # Select regions to adjust
    selected_regions = ["Left Cheek", "Right Cheek", "Upper Lip", "Lower Lip", "Left Eye", "Right Eye", "Left Eyebrow", "Right Eyebrow", "Nose", "Forehead", "Chin", "Jawline"]  # Modify as needed

    # Create combined mask for selected regions
    combined_mask = None
    for region in selected_regions:
        indices = regions[region]
        mask = create_facial_region_mask(image, landmarks, indices)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Set adjustment parameters
    hue_shift = 10          # Adjust as needed
    saturation_scale = 1.2  # Adjust as needed
    lightness_scale = 1.0   # Adjust as needed

    # Adjust skin color on selected regions
    adjusted_image = adjust_skin_color(
        image,
        hue_shift=hue_shift,
        saturation_scale=saturation_scale,
        lightness_scale=lightness_scale,
        region_mask=combined_mask
    )

    # Display the original and adjusted images side by side
    combined_display = np.hstack((image, adjusted_image))
    cv2.imshow('Original (Left) vs Adjusted (Right)', combined_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the adjusted image
    save_path = 'adjusted_image.jpg'
    cv2.imwrite(save_path, adjusted_image)
    print(f"Adjusted image saved as {save_path}")

if __name__ == "__main__":
    main()

