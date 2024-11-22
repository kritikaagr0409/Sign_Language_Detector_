# Hand Gesture Data Processing with Mediapipe

This project processes images of hand gestures, extracts meaningful features using Mediapipe, and stores the features and labels in a serialized format (`data.pickle`) for training machine learning models.

---

## Features

- **Data Organization**: Reads gesture images from a directory where each subdirectory represents a gesture class.
- **Feature Extraction**: Uses Mediapipe's hand landmark detection to extract features.
- **Data Serialization**: Saves the extracted features and their corresponding labels in a pickle file for further use.

---

## Prerequisites

### Libraries and Tools

Ensure you have the following installed:

- Python 3.6+
- OpenCV
- Mediapipe

Install the required packages using:
```bash
pip install opencv-python mediapipe
