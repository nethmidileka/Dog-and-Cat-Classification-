import cv2
import numpy as np
import pyttsx3
import joblib
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys

# Load the trained model
svm_model = joblib.load('svm_model.pkl')

# Function to classify an image
def classify_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (50, 50))  # Resize image to 50x50 pixels
    flattened_img = resized_img.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    prediction = svm_model.predict(flattened_img)
    if prediction[0] == 1:
        return "Cat"
    elif prediction[0] == 0:
        return "Dog"
    else:
        return "Unknown"

# Function to display result in voice
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Create a simple GUI
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Cat VS Dog")
        self.setGeometry(100, 100, 640, 480)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(10, 10, 620, 420)

        self.btn = QPushButton("Upload Image", self)
        self.btn.setGeometry(250, 440, 140, 30)
        self.btn.clicked.connect(self.open_file_dialog)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn)
        self.setLayout(self.layout)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self,"Open Image File", "","All Files (*);;Image Files (*.png *.jpg *.jpeg)", options=options)
        if filename:
            img = cv2.imread(filename)
            result = classify_image(img)
            
            speak(result)
            print(result)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (620, 420))
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
            self.label.setText(result)

 

# Start the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())

