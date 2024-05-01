# Face Recognition using Machine Learning with OpenCV and Python

Welcome to the Face Recognition project using Machine Learning, OpenCV, and Python! This project aims to demonstrate how to build a face recognition system using popular libraries and tools. Below you will find detailed instructions on how to set up and run this project.

## Installation

To install the necessary libraries and dependencies for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/sriramsowmithri9807/face_recognition.git
   ```

2. Install required libraries using pip:
   ```bash
   pip install numpy opencv-python scikit-learn
   ```

3. Install additional libraries for face recognition:
   ```bash
   pip install face_recognition
   ```

4. Download the pre-trained model for face detection:
   ```bash
   wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
   ```

## Usage

1. Prepare your dataset:
   - Create a folder named `dataset` and add subfolders for each person with their images.
   
2. Train the model:
   - Run the `train_model.py` script to train the face recognition model on your dataset.

3. Test the model:
   - Use the `recognize_faces.py` script to test the trained model on new images.

4. Enjoy face recognition:
   - Have fun recognizing faces using your machine learning model!

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes. Your contributions are greatly appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Thank you for checking out this Face Recognition project! If you have any questions or feedback, please feel free to reach out. Happy coding! 🚀
