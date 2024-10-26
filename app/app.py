from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import os
import time
import json
from itertools import combinations


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # アップロードフォルダをstaticに指定
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # フォルダが存在しない場合は作成

db = SQLAlchemy(app)

class ImageData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)

db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/approad', methods=['GET', 'POST'])
def approad():
    if request.method == 'POST':
        # 画像アップロードの処理
        if 'image' not in request.files:
            return redirect(url_for('approad'))

        image_file = request.files['image']
        if image_file.filename == '':
            return redirect(url_for('approad'))

        # アップロードされた画像を保存
        image_filename = f"{int(time.time())}_{image_file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)

        # operationページに画像パスを渡してリダイレクト
        return redirect(url_for('operation', image_path=image_file.filename))

    return render_template('approad.html')

@app.route('/operation', methods=['GET', 'POST'])
def operation():
    image_path = request.args.get('image_path')
    recommendations = []
    if request.method == 'POST':
        current_tone = {
            "色相": int(request.form['currentHue']),
            "彩度": int(request.form['currentSaturation']),
            "明度": int(request.form['currentLightness'])
        }
        target_tone = {
            "色相": int(request.form['targetHue']),
            "彩度": int(request.form['targetSaturation']),
            "明度": int(request.form['targetLightness'])
        }

        # AJAXを通じて化粧品の提案を取得
        recommendations = get_cosmetic_recommendations(current_tone, target_tone)

        # ここでリダイレクト
        return redirect(url_for('result', image_path=image_path, recommendations=json.dumps(recommendations)))

    return render_template('operation.html', image_path=image_path)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No file uploaded.'}), 400

    image_file = request.files['image']
    hue_shift = int(request.form['hue_shift'])
    saturation_scale = float(request.form['saturation_scale'])
    lightness_scale = float(request.form['lightness_scale'])

    # アップロードされた画像を一時的に保存
    timestamp = int(time.time())
    image_filename = f"{timestamp}_{image_file.filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    image_file.save(image_path)

    # 元の画像の平均色相、彩度、明度を取得
    original_image = cv2.imread(image_path)
    original_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    original_hue = np.mean(original_hsv[:, :, 0])  # 平均色相
    original_saturation = np.mean(original_hsv[:, :, 1])  # 平均彩度
    original_lightness = np.mean(original_hsv[:, :, 2])  # 平均明度

    # Processing the image
    try:
        adjusted_image = adjust_skin_color(image_path, hue_shift, saturation_scale, lightness_scale)
        if adjusted_image is None:
            raise ValueError("Image processing failed.")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # 画像を処理して調整された画像のパスを返す
    adjusted_image = adjust_skin_color(image_path, hue_shift, saturation_scale, lightness_scale)

    if adjusted_image is None:
        return jsonify({'error': 'Image processing failed.'}), 500

    # 結果画像を保存してパスを返す
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'adjusted_image_{timestamp}.jpg')  # 保存先を変更
    cv2.imwrite(result_image_path, adjusted_image)

    # 操作後の色相、彩度、明度を計算
    adjusted_hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV)
    adjusted_hue = np.mean(adjusted_hsv[:, :, 0])
    adjusted_saturation = np.mean(adjusted_hsv[:, :, 1])
    adjusted_lightness = np.mean(adjusted_hsv[:, :, 2])

    # 相対的な変化量を計算
    hue_change = adjusted_hue - original_hue
    saturation_change = adjusted_saturation - original_saturation
    lightness_change = adjusted_lightness - original_lightness

    return jsonify({
        'image_path': url_for('static', filename=f'uploads/adjusted_image_{timestamp}.jpg'),
        'hue_change': hue_change,
        'saturation_change': saturation_change,
        'lightness_change': lightness_change
    })



def adjust_skin_color(image_path, hue_shift=0, saturation_scale=1.0, lightness_scale=1.0):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

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

    return result

@app.route('/get_cosmetic_recommendations', methods=['POST'])
def get_cosmetic_recommendations():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided.'}), 400

    current_tone = data.get('current_tone')
    target_tone = data.get('target_tone')

    if not current_tone or not target_tone:
        return jsonify({'error': 'Missing tone data.'}), 400

    # 化粧品データベースの読み込み
    try:
        with open("cosmetics_database.json", "r", encoding="utf-8") as f:
            cosmetics_data = json.load(f)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # DPテーブルの初期化
    dp = {}
    initial_state = tuple(current_tone.values())
    dp[initial_state] = (float("inf"), None)

    # 商品タイプのリストを作成
    product_types = set(item["type"] for item in cosmetics_data)

    # 遷移を行う関数
    def update_dp(selected_products):
        new_tone = {
            feature: current_tone[feature]
            + sum(product[feature] for product in selected_products)
            for feature in target_tone
        }
        diff = sum((new_tone[feature] - target_tone[feature]) ** 2 for feature in target_tone)
        new_state = tuple(new_tone.values())
        if new_state not in dp or dp[new_state][0] > diff:
            dp[new_state] = (diff, [product["name"] for product in selected_products])

    # 各商品タイプごとに商品をグループ化
    grouped_products = {ptype: [] for ptype in product_types}
    for product in cosmetics_data:
        grouped_products[product["type"]].append(product)

    # 組み合わせの生成とDP更新
    for r in range(1, len(cosmetics_data) + 1):
        for product_combination in combinations(cosmetics_data, r):
            if len(set(p["type"] for p in product_combination)) == len(product_combination):
                update_dp(product_combination)

    # 最適解の探索
    best_state = min(dp, key=lambda x: dp[x][0])
    recommendations = dp[best_state][1] if dp[best_state][1] else []

    return jsonify(recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
