import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5)

def get_facial_landmarks(image):
    """
    Detects facial landmarks in an image using MediaPipe.
    Returns a list of (x, y) tuples representing landmark positions.
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    # Get the first face detected
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
    
    Args:
        image: Original BGR image.
        landmarks: List of (x, y) tuples.
        region_indices: List of indices corresponding to the desired facial region.
    
    Returns:
        Mask image with the specified region filled white and others black.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = [landmarks[i] for i in region_indices]
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def adjust_skin_color(image, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0, region_mask=None):
    """
    Adjusts the skin tone of specified facial regions in an image.
    
    Args:
        image: Original BGR image.
        hue_shift: Integer, hue shift value (-179 to 179).
        saturation_scale: Float, saturation scaling factor.
        lightness_scale: Float, lightness scaling factor.
        region_mask: Optional mask to specify regions to adjust.
    
    Returns:
        Adjusted image.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # If a region mask is provided, use it; else, adjust entire skin
    if region_mask is not None:
        skin_mask = region_mask
    else:
        # Define skin color range in HSV
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    
    # Extract skin regions
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    
    # Convert skin to HSV for adjustments
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(skin_hsv)
    
    # Adjust Hue
    h_channel = np.mod(h_channel.astype(np.int32) + hue_shift, 180).astype(np.uint8)
    
    # Adjust Saturation
    s_channel = cv2.multiply(s_channel, saturation_scale)
    s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)
    
    # Adjust Lightness (Value)
    v_channel = cv2.multiply(v_channel, lightness_scale)
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)
    
    # Merge channels back
    adjusted_hsv = cv2.merge((h_channel, s_channel, v_channel))
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
    
    # Define regions by landmark indices
    # MediaPipe's Face Mesh has 468 landmarks; refer to the documentation for exact indices
    # Here's an example for the cheeks:
    left_cheek_indices = list(range(234, 247))  # Example indices for left cheek
    right_cheek_indices = list(range(454, 467)) # Example indices for right cheek
    
    # Define regions for lips (optional)
    upper_lip_indices = list(range(13, 15))    # Example indices for upper lip
    lower_lip_indices = list(range(78, 80))    # Example indices for lower lip
    
    # Create masks for the cheeks
    left_cheek_mask = create_facial_region_mask(image, landmarks, left_cheek_indices)
    right_cheek_mask = create_facial_region_mask(image, landmarks, right_cheek_indices)
    
    # Combine masks if you want to adjust both cheeks simultaneously
    combined_cheek_mask = cv2.bitwise_or(left_cheek_mask, right_cheek_mask)
    
    # Similarly, create masks for other regions if needed
    # For example, creating a mask for upper and lower lips
    upper_lip_mask = create_facial_region_mask(image, landmarks, upper_lip_indices)
    lower_lip_mask = create_facial_region_mask(image, landmarks, lower_lip_indices)
    combined_lip_mask = cv2.bitwise_or(upper_lip_mask, lower_lip_mask)
    
    # Decide which regions to adjust
    # For this example, let's adjust only the cheeks
    regions_to_adjust = combined_cheek_mask  # You can combine multiple masks if needed
    
    # Adjust skin tone only on the selected regions
    adjusted_image = adjust_skin_color(
        image,
        hue_shift=10,              # Adjust as needed
        saturation_scale=1.2,      # Adjust as needed
        lightness_scale=1.0,       # Adjust as needed
        region_mask=regions_to_adjust
    )
    
    # Display the original and adjusted images side by side
    combined_display = np.hstack((image, adjusted_image))
    cv2.imshow('Original Image (Left) vs Adjusted Skin Color (Right)', combined_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally, save the adjusted image
    # cv2.imwrite('adjusted_image.jpg', adjusted_image)

if __name__ == "__main__":
    main()
