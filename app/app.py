from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import os
import time
import json
from itertools import combinations
import mediapipe as mp

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

class ImageData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)

db.create_all()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/approad', methods=['GET', 'POST'])
def approad():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(url_for('approad'))

        image_file = request.files['image']
        if image_file.filename == '':
            return redirect(url_for('approad'))

        image_filename = f"{int(time.time())}_{image_file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)

        return redirect(url_for('operation', image_path=image_filename))

    return render_template('approad.html')

@app.route('/operation', methods=['GET', 'POST'])
def operation():
    image_path = request.args.get('image_path')
    recommendations = []
    if request.method == 'POST':
        hue_shift = int(request.form['hue_shift'])
        saturation_scale = float(request.form['saturation_scale'])
        lightness_scale = float(request.form['lightness_scale'])

        recommendations = get_cosmetic_recommendations(request.form)

        image_full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
        original_image = cv2.imread(image_full_path)
        adjusted_image = process_face_segmentation(original_image, hue_shift, saturation_scale, lightness_scale)

        if adjusted_image is None:
            return jsonify({'error': 'Image processing failed.'}), 500

        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'adjusted_image_{int(time.time())}.jpg')
        cv2.imwrite(result_image_path, adjusted_image)

        return redirect(url_for('result', image_path=os.path.basename(result_image_path), recommendations=json.dumps(recommendations)))

    return render_template('operation.html', image_path=image_path)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No file uploaded.'}), 400

    image_file = request.files['image']
    hue_shift = int(request.form['hue_shift'])
    saturation_scale = float(request.form['saturation_scale'])
    lightness_scale = float(request.form['lightness_scale'])

    timestamp = int(time.time())
    image_filename = f"{timestamp}_{image_file.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    original_image = cv2.imread(image_path)
    adjusted_image = process_face_segmentation(original_image, hue_shift, saturation_scale, lightness_scale)

    if adjusted_image is None:
        return jsonify({'error': 'Image processing failed.'}), 500

    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'adjusted_image_{timestamp}.jpg')
    cv2.imwrite(result_image_path, adjusted_image)

    return jsonify({
        'image_path': url_for('static', filename=f'uploads/adjusted_image_{timestamp}.jpg')
    })

# 肌色調整のための関数
def adjust_skin_color(face, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0):
    # HSV色空間に変換し、色相、彩度、明度を調整
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    
    # 肌色の範囲を定義（HSV色空間）
    lower_skin = np.array([0, 20, 40], dtype=np.uint8)  # 色相、彩度、明度
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)  # 色相、彩度、明度

    # 肌色領域のマスクを作成
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # マスクを使って肌色領域を抽出
    skin = cv2.bitwise_and(face, face, mask=skin_mask)

    # HSV色空間で色調整
    skin_hsv = cv2.cvtColor(skin, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(skin_hsv)

    h_channel = np.mod(h_channel.astype(np.int32) + hue_shift, 180).astype(np.uint8)
    s_channel = np.clip(cv2.multiply(s_channel, saturation_scale), 0, 255).astype(np.uint8)
    v_channel = np.clip(cv2.multiply(v_channel, lightness_scale), 0, 255).astype(np.uint8)

    adjusted_hsv = cv2.merge((h_channel, s_channel, v_channel))
    adjusted_skin = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_skin

def process_face_segmentation(image, hue_shift, saturation_scale, lightness_scale):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # マスクを作成して顔の部分を抽出
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks.landmark]
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(mask, hull, 255)

        # 顔の部分を抽出
        face = cv2.bitwise_and(image, image, mask=mask)
        
        # デバッグ出力
        print("Face Shape:", face.shape)

        # 肌色を調整
        processed_face = adjust_skin_color(face, hue_shift, saturation_scale, lightness_scale)

        # 元の画像から背景を抽出
        inverse_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(image, image, mask=inverse_mask)

        # 調整された顔の部分と元の背景を結合
        result = cv2.add(background, processed_face)

        return result
    else:
        print("Error: No face detected.")
        return None


@app.route('/get_cosmetic_recommendations', methods=['POST'])
def get_cosmetic_recommendations():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided.'}), 400

    current_tone = data.get('current_tone')
    target_tone = data.get('target_tone')

    if not current_tone or not target_tone:
        return jsonify({'error': 'Missing tone data.'}), 400

    try:
        with open("cosmetics_database.json", "r", encoding="utf-8") as f:
            cosmetics_data = json.load(f)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    dp = {}
    initial_state = tuple(current_tone.values())
    dp[initial_state] = (float("inf"), None)

    product_types = set(item["type"] for item in cosmetics_data)

    def update_dp(selected_products):
        new_tone = {
            feature: current_tone[feature]
            + sum(product[feature] for product in selected_products)
            for feature in target_tone
        }
        diff = sum((abs(new_tone[feature]) - abs(target_tone[feature])) ** 2 for feature in target_tone)
        new_state = tuple(new_tone.values())
        if new_state not in dp or dp[new_state][0] > diff:
            dp[new_state] = (diff, [product["name"] for product in selected_products])

    grouped_products = {ptype: [] for ptype in product_types}
    for product in cosmetics_data:
        grouped_products[product["type"]].append(product)

    for r in range(1, len(cosmetics_data) + 1):
        for product_combination in combinations(cosmetics_data, r):
            if len(set(p["type"] for p in product_combination)) == len(product_combination):
                update_dp(product_combination)

    best_state = min(dp, key=lambda x: dp[x][0])
    recommendations = dp[best_state][1] if dp[best_state][1] else []

    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)