import cv2
import numpy as np

def adjust_skin_color(image_path, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0):
    hue_shift = max(-10, min(10, hue_shift))
    saturation_scale = max(0.8, min(1.2, saturation_scale))
    lightness_scale = max(0.8, min(1.2, lightness_scale))

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 180, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(skin_hsv)

    h_channel = np.mod(h_channel.astype(np.int32) + hue_shift, 180).astype(np.uint8)
    s_channel = cv2.multiply(s_channel, saturation_scale)
    s_channel = np.clip(s_channel, 0, 255).astype(np.uint8)
    v_channel = cv2.multiply(v_channel, lightness_scale)
    v_channel = np.clip(v_channel, 0, 255).astype(np.uint8)

    adjusted_hsv = cv2.merge((h_channel, s_channel, v_channel))
    adjusted_skin = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    inverse_skin_mask = cv2.bitwise_not(skin_mask)
    background = cv2.bitwise_and(image, image, mask=inverse_skin_mask)
    result = cv2.add(background, adjusted_skin)

    combined = np.hstack((image, result))
    cv2.imshow('Original Image (Left) vs Adjusted Skin Color (Right)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用例
adjust_skin_color('path_to_your_image.jpg', hue_shift=5, saturation_scale=1.1, lightness_scale=1.0)
