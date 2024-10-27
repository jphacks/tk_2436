import cv2
import numpy as np

def adjust_skin_color(image_path, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjusted skin color range in HSV
    lower_skin = np.array([20, 10, 20], dtype=np.uint8)
    upper_skin = np.array([45, 50, 95], dtype=np.uint8)

    # Create a mask for skin regions
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # Blur the mask to reduce noise
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # Extract skin regions from the original image
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Convert skin region to HSV color space for color adjustment
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

    # Merge channels and convert back to BGR color space
    adjusted_hsv = cv2.merge((h_channel, s_channel, v_channel))
    adjusted_skin = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    # Combine the adjusted skin back into the original image
    inverse_skin_mask = cv2.bitwise_not(skin_mask)
    background = cv2.bitwise_and(image, image, mask=inverse_skin_mask)
    result = cv2.add(background, adjusted_skin)

    # Display the original and adjusted images side by side
    combined = np.hstack((image, result))
    cv2.imshow('Original Image (Left) vs Adjusted Skin Color (Right)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
# Adjust the hue_shift (-179 to 179), saturation_scale (0.0 to 3.0), and lightness_scale (0.0 to 3.0)
adjust_skin_color('path_to_your_image.jpg', hue_shift=10, saturation_scale=1.2, lightness_scale=1.0)
