from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import os
import time

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
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # operationページに画像パスを渡してリダイレクト
        return redirect(url_for('operation', image_path=image_file.filename))

    return render_template('approad.html')

@app.route('/operation', methods=['GET', 'POST'])
def operation():
    image_path = request.args.get('image_path')
    if request.method == 'POST':
        return redirect(url_for('result'))

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

    # 画像を処理して調整された画像のパスを返す
    adjusted_image = adjust_skin_color(image_path, hue_shift, saturation_scale, lightness_scale)

    if adjusted_image is None:
        return jsonify({'error': 'Image processing failed.'}), 500

    # 結果画像を保存してパスを返す
    result_image_path = os.path.join('static', f'adjusted_image_{timestamp}.jpg')
    cv2.imwrite(result_image_path, adjusted_image)

    return jsonify({'image_path': url_for('static', filename=f'adjusted_image_{timestamp}.jpg')})

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

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
