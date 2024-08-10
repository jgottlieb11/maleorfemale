# Gender Recognition System

This project is a real-time facial recognition system designed to identify gender using a pre-trained model with the DeepFace library. The system is built in Python, leveraging various tools and libraries like OpenCV, Pandas, and Matplotlib for face detection, data management, and performance visualization.

## Features

- **Real-Time Recognition**: Processes video feeds to detect and classify gender in real-time.
- **DeepFace Integration**: Uses DeepFace for feature extraction from facial images.
- **SVM Classifier**: Trained on the CelebA dataset from Kaggle to accurately classify gender.
- **Performance Metrics**: Evaluates model performance using accuracy and F1-score, with results visualized through confusion matrices.

## Installation

To set up and run the system, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/jgottlieb11/gender-recognition-system.git
    cd gender-recognition-system
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the system**:
    ```bash
    python real_time_recognition.py
    ```

## Usage

- **Real-Time Recognition**: The system captures video feeds via webcam and classifies detected faces as "Male" or "Female" based on the trained model.
- **Evaluation**: The `evaluate_model_performance.py` script allows you to evaluate the system's accuracy and F1-score on a subset of the CelebA dataset, with results visualized using confusion matrices.
- **Test on .jpg**: The `test_model.py` script allows you to evaluate the system by choosing an image within img_align_celeba, with the result being either male or female.


## Dataset

The system is trained and evaluated on the CelebA dataset, which can be found on Kaggle [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Acknowledgments

- **DeepFace Library**: For providing tools for facial recognition and analysis.
- **Kaggle**: For hosting the CelebA dataset used in this project.

