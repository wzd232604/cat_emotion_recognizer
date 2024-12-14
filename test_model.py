import torchaudio
import torch
from torch.nn.functional import softmax

# 定义模型结构（与训练时保持一致）
class AudioClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128 * (16000 // 4), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 加载模型
model_path = 'best_audio_classifier.pth'  # 替换为你的模型路径
num_classes = 3  # 类别数量
model = AudioClassifier(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 音频预处理函数
def preprocess_audio(filepath, fixed_length=16000):
    signal, sr = torchaudio.load(filepath)
    if signal.size(0) > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)  # 转为单通道
    if signal.size(1) < fixed_length:
        signal = torch.nn.functional.pad(signal, (0, fixed_length - signal.size(1)))  # 零填充
    else:
        signal = signal[:, :fixed_length]  # 截断
    return signal

# 测试函数
def test_model(filepath):
    # 预处理音频
    input_signal = preprocess_audio(filepath)
    input_tensor = input_signal.unsqueeze(0)  # 增加 batch 维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

    # 映射到情绪类别
    emotions = ["我要梳毛", "我要吃饭", "我要抱抱"]
    print(f"音频文件: {filepath}")
    print(f"预测情绪: {emotions[predicted_class]}")
    print(f"概率分布: {probabilities.tolist()}")

# 固定测试文件路径
if __name__ == "__main__":
    test_audio_path = '../cat-voice/dataset/B_ANI01_MC_FN_SIM01_101.wav'  # 替换为实际的音频文件路径
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        print(f"正在测试音频文件: {test_audio_path}")
        test_model(test_audio_path)
