# Dog-and-Cat-Classification
Machine learning project

Object Recognition Application
-----------------------------------
This application allows users to upload an image and performs object recognition using a pre-trained machine learning model.
The model is trained to classify images into two categories: "Dog" and "Cat".

Features
----------
Upload an image from your computer
Display the uploaded image along with the predicted class label (either "Dog" or "Cat")


Requirements
------------------
1.Python 3.x

2.PyQt5

3.OpenCV (cv2)

4.scikit-learn (sklearn)

5.joblib

Notes
---------
1.The application uses a Support Vector Machine (SVM) model trained on grayscale images resized to 50x50 pixels.

2.The model assigns class labels: 0 for "Dog" and 1 for "Cat".

3.If the model predicts an incorrect class label, it may be due to limitations in the training data or the complexity of the image.

Contributions
------------
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.


License
----------
This project is licensed under the MIT License - see the LICENSE file for details.
