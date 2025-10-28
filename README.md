✍️ Real-Time Air Drawing Digit Recognition (MNIST)

This project implements a real-time system for recognizing handwritten digits drawn in the air using an index finger. It leverages **Google's MediaPipe** for hand tracking, **OpenCV** for image processing, and three different machine learning models (**KNN, SVM, and ANN**) trained on the **MNIST** dataset to perform live classification.

The application is designed to run primarily in a **Google Colab** environment for easy access to the necessary dependencies and camera stream integration.

✨ Features

  * **Air Drawing:** Use your index finger to draw digits (0-9) in the camera's view.
  * **Real-Time Hand Tracking:** Utilizes MediaPipe's Hand Landmarks model to accurately track the index finger position.
  * **Multi-Model Comparison:** Simultaneously predicts the drawn digit using three pre-trained models:
      * **K-Nearest Neighbors (KNN)**
      * **Support Vector Machine (SVM)**
      * **Artificial Neural Network (ANN)**
  * **Digit Preprocessing:** Automatically crops, squares, and resizes the drawn digit to the $28 \times 28$ pixel format required by the MNIST models.
  * **Live Confidence Scores:** Displays the prediction and confidence score for each model in real time.


🛠️ Prerequisites

This project is best executed within **Google Colab**.

The code requires the following Python libraries, which are installed in the first cell:

```bash
!pip install -q scikit-learn tensorflow numpy matplotlib opencv-python mediapipe pillow
```

🚀 How to Run

1.  **Open in Google Colab:** Click the "Open in Colab" badge (if you create one) or copy the code into a new Colab notebook.
2.  **Run All Cells:** Execute all the code cells in the notebook sequentially.
3.  **Grant Camera Permission:** When prompted, grant access to your webcam.
4.  **Draw:** Once the video stream starts, show your hand and use your **index finger** to draw a digit in the air. The drawing will appear on the screen.
5.  **View Predictions:** The real-time predictions from the KNN, SVM, and ANN models, along with their confidence, will be displayed below the video feed.
6.  **Clear Canvas:** Press the **'C'** key on your keyboard to clear the drawing canvas and start a new digit.
7.  **Stop:** Click directly on the video feed to stop the application.


🧠 Model Architectures

The system uses three distinct models trained on subsets of the MNIST dataset for classification:

### 1\. K-Nearest Neighbors (KNN)

  * **Dataset:** Subset of 10,000 training samples.
  * **Hyperparameters:** $n\_neighbors = 3$.

### 2\. Support Vector Machine (SVM)

  * **Dataset:** Subset of 10,000 training samples.
  * **Hyperparameters:** `kernel='rbf'`, `C=10`, `gamma='scale'`.

### 3\. Artificial Neural Network (ANN)

  * **Dataset:** Full 60,000 training samples.
  * **Architecture (Multi-Layer Perceptron):**
      * Input Layer: 784 features.
      * Hidden Layer 1: 128 neurons, ReLU activation.
      * Dropout (0.2).
      * Hidden Layer 2: 64 neurons, ReLU activation.
      * Dropout (0.2).
      * Output Layer: 10 neurons (0-9), Softmax activation.
  * **Training:** 10 epochs, `batch_size=128`.

⚙️ Key Code Functions

| Function | Description |
| :--- | :--- |
| `js_to_image(js_reply)` | Converts the base64 encoded image data from the JavaScript camera stream into an OpenCV (NumPy array) image format. |
| `video_stream()`/`video_frame()` | Custom functions using `IPython.display` and `eval_js` to establish and maintain the real-time video stream in the Colab environment. |
| `preprocess_for_prediction(canvas)` | **Crucial step\!** Takes the full drawing canvas, finds the digit contour, crops the **Region of Interest (ROI)**, centers it, resizes it to $28 \times 28$, normalizes it, and flattens it into the $1 \times 784$ vector required by the models. |
| `predict_digit(processed_input)` | Takes the preprocessed $1 \times 784$ input vector and passes it through the trained KNN, SVM, and ANN models to get predictions and confidence scores. |

🛑 How to Stop

To ensure the camera stream is properly released, click on the active video feed inside the Colab output. The Python kernel will catch the shutdown signal, and the code will execute the `finally` block, printing **"SESSION ENDED"**.

-----

## 🤝 Contributing

Feel free to fork the repository and submit pull requests. Any improvements to hand-tracking robustness, digit preprocessing, or model performance are welcome\!
