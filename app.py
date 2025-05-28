
import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ────────────────────────────────────────────────
# Flask + SocketIO setup
# ────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet',  cors_allowed_origins="*")

# ────────────────────────────────────────────────
# EmotionCNN model definition
# ────────────────────────────────────────────────
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ELU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 64), nn.ELU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(64, 64), nn.ELU(), nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(64, 7)  # <- 7 classes as per your trained model
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ────────────────────────────────────────────────
# Load model and face detector
# ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("Emotion_little_vgg.pth", map_location=device))
model.eval()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# ────────────────────────────────────────────────
# Socket.IO event handlers
# ────────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("connection_response", {"message": "Connected to server"})

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

@socketio.on("frame")
def handle_frame(data):
    try:
        image_data = data["frame"].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        """
        if frame is None or frame.size == 0:
            print("⚠️ Empty or invalid frame received")
            emit("emotion", {"emotion": "no frame"})
            return
        """

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=5)
        faces = face_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.1,       # Try 1.1 instead of 1.3
            minNeighbors=4,        # Lower means more detections
            minSize=(60, 60)       # Helps filter tiny false positivesw
        )

        # cv2.imwrite("debug_frame.jpg", frame)

        if len(faces) == 0:
            emotion = "no face"
        else:
            # Take the first face found
            x, y, w, h = faces[0]
            face = grayscale[y:y+h, x:x+w]
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                pred = output.argmax(1).item()
                emotion = classes[pred]

    except Exception as e:
        print(f"Emotion detection error: {e}")
        emotion = "error"

    emit("emotion", {"emotion": emotion})

# ────────────────────────────────────────────────
# Run server
# ────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app)
    #socketio.run(app, host="0.0.0.0", port=5000, debug=True)
