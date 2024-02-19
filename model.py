import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Function to extract features from images
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (50, 50))
    flattened_img = resized_img.flatten()
    return flattened_img

# Path to the directory 
data_dir = r"C:\Users\nethm\OneDrive\Desktop\dog vs cat\train"
cat_dir = os.path.join(data_dir, "cats")
dog_dir = os.path.join(data_dir, "dogs")

# Load images and extract features
X = []
y = []

# Load cat images
for img_name in os.listdir(cat_dir):
    img_path = os.path.join(cat_dir, img_name)
    features = extract_features(img_path)
    X.append(features)
    y.append(1)  # Label 1 for cat

# Load dog images
for img_name in os.listdir(dog_dir):
    img_path = os.path.join(dog_dir, img_name)
    features = extract_features(img_path)
    X.append(features)
    y.append(0)  # Label 0 for dog

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(svm_model, 'svm_model.pkl')

# Get the label numbers
label_numbers = svm_model.classes_

# Display the label numbers
print("Label Numbers:", label_numbers)
