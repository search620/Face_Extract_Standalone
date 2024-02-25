# Face_Extract_Standalone


This repository showcases an advanced face detection and recognition system built using the FaceNet PyTorch framework, MTCNN for face detection, and InceptionResnetV1 for generating face embeddings. It's designed to process images and videos, detect faces with high accuracy, and recognize faces by comparing them to a target face. Here are the highlights:

**High-Performance Face Detection:** Utilizes MTCNN for detecting faces in images and videos with high accuracy.

**Advanced Face Recognition:** Employs InceptionResnetV1, pretrained on the 'vggface2' dataset, to generate face embeddings for recognition purposes.

**Flexible Image Processing:** Supports processing of both images and videos, with options for resizing, sharpness checks, and exporting full frames or cropped faces.

**Customizable Parameters:** Offers a wide range of customizable parameters, including batch size, resize scale, sharpness threshold, and more, to optimize performance and accuracy according to specific needs.
GPU Acceleration: Leverages GPU acceleration (if available) for fast and efficient processing.
Logging and Progress Tracking: Includes optional logging for debugging and progress tracking with tqdm for a better user experience.
This system is perfect for applications requiring reliable face detection and recognition.
