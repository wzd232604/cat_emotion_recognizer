from flask import Flask, request, render_template, redirect, url_for
import librosa
import torch

app = Flask(__name__)

# 模型加载
# model_path = './model/cat_emotion_model.pth'
# model = torch.load(model_path)
# model.eval()

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 情绪识别 API
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('audioFile')
    if file:
        # 保存上传文件
        filepath = f'./uploads/{file.filename}'
        file.save(filepath)

        # 处理音频文件
        y, sr = librosa.load(filepath, sr=16000)
        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=40)
        input_tensor = torch.tensor(mfccs).unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            output = model(input_tensor)
            emotion = torch.argmax(output).item()

        # 将识别的情绪传递给结果页面
        emotions = ["我要梳毛", "我要吃饭", "我要抱抱"]
        return render_template('result.html', emotion=emotions[emotion])

    return redirect(url_for('index'))  # 如果没有文件，重定向到首页

if __name__ == '__main__':
    app.run(debug=True)
