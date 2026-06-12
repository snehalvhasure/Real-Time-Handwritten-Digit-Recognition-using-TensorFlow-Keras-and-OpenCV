import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

# ==========================
# Load MNIST Dataset
# ==========================
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ==========================
# Scaling
# ==========================
X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

# ==========================
# Model Building
# ==========================
seq_model = Sequential()

seq_model.add(Input(shape=(28, 28)))
seq_model.add(Flatten())

seq_model.add(Dense(128, activation='relu'))
seq_model.add(Dense(256, activation='relu'))
seq_model.add(Dense(512, activation='relu'))
seq_model.add(Dense(256, activation='relu'))
seq_model.add(Dense(128, activation='relu'))

seq_model.add(Dense(10, activation='softmax'))

seq_model.summary()

# ==========================
# Compile
# ==========================
seq_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train
# ==========================
seq_model.fit(
    X_train_scale,
    y_train,
    epochs=10
)

# ==========================
# Evaluate
# ==========================
seq_model.evaluate(
    X_test_scale,
    y_test
)

# ==========================
# CCTV / Camera Prediction
# ==========================

cap = cv2.VideoCapture(0)   # 0 = default webcam

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to MNIST size
    digit_img = cv2.resize(gray, (28, 28))

    # Normalize
    digit_img = digit_img / 255.0

    # Reshape for model
    digit_img = digit_img.reshape(1, 28, 28)

    # Prediction
    prediction = seq_model.predict(digit_img, verbose=0)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    # Display result
    cv2.putText(
        frame,
        f"Digit: {digit} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Digit Recognition CCTV", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()