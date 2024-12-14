from flask import Flask, request, render_template, redirect, url_for
import os
import torchaudio
import torch
from torch.nn.functional import softmax

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * (16000 // 4), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load pre-trained model
model_path = 'best_audio_classifier.pth'
num_classes = 3  # Assuming three classes: "我要梳毛", "我要吃饭", "我要抱抱"
model = AudioClassifier(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Audio preprocessing
def preprocess_audio(filepath, fixed_length=16000):
    signal, sr = torchaudio.load(filepath)
    if signal.size(0) > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)  # Convert to mono

    # Pad or truncate to fixed length
    if signal.size(1) < fixed_length:
        signal = torch.nn.functional.pad(signal, (0, fixed_length - signal.size(1)))
    else:
        signal = signal[:, :fixed_length]
    
    return signal

# Homepage route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('audioFile')
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess audio
        input_signal = preprocess_audio(filepath)
        input_tensor = input_signal.unsqueeze(0)  # Add batch dimension

        # Model inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = softmax(output, dim=1)
            emotion = torch.argmax(probabilities).item()

        # Map prediction to emotion
        emotions = ["我要梳毛", "我要吃饭", "我要抱抱"]
        return render_template('result.html', emotion=emotions[emotion])

    return redirect(url_for('index'))  # Redirect to homepage if no file is uploaded

if __name__ == '__main__':
    app.run(debug=True)
