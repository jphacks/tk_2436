from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class ImageData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(150), nullable=False)
    # 他の必要なフィールドを追加

db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/approad', methods=['GET', 'POST'])
def approad():
    if request.method == 'POST':
        # アップロードされた画像を処理するロジックをここに追加
        return redirect(url_for('operation'))
    return render_template('approad.html')

@app.route('/operation', methods=['GET', 'POST'])
def operation():
    if request.method == 'POST':
        # 画像を操作し、結果を取得するロジックをここに追加
        return redirect(url_for('result'))
    return render_template('operation.html')

@app.route('/result')
def result():
    # データベースからデータを取得するロジックをここに追加
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
