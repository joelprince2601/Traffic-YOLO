Traffic-YOLO
Traffic-YOLO is an advanced real-time object detection and tracking system designed to improve traffic management and analysis. By combining YOLO (You Only Look Once) for efficient object detection with DeepSort for robust tracking, this project offers a powerful tool for monitoring traffic scenarios.

Features
Real-time Object Detection: Detects and identifies traffic-related objects such as vehicles and pedestrians in real-time using YOLO.
Object Tracking: Maintains consistent tracking of detected objects across video frames with DeepSort.
Distance Measurement: Displays the distance to detected objects on the video frame for enhanced situational awareness.
Leader Lines: Visualizes leader lines pointing towards destinations while avoiding obstacles, aiding in navigation and traffic flow analysis.
Installation
To get started with Traffic-YOLO, follow these steps:

Clone the Repository:

bash
Copy code
git clone https://github.com/joelprince2601/Traffic-YOLO.git
cd Traffic-YOLO
Install Dependencies:

Make sure you have Python 3.7 or higher installed. Then, install the required Python packages using:

bash
Copy code
pip install -r requirements.txt
Download YOLO Weights:

Download the YOLO weights file from YOLO's official website and place it in the project directory.

Download DeepSort Model:

Obtain the DeepSort model weights from DeepSort's repository.

Usage
Run the Detection and Tracking Script:

Execute the following command to start the object detection and tracking system on your video feed:

bash
Copy code
python main.py --video <path_to_video_file>
Replace <path_to_video_file> with the path to your video file or use --source 0 to use your webcam.

Adjust Configuration:

Modify the config.py file to adjust settings such as detection thresholds, model paths, and other parameters.

Example
Here is an example command to run the script with a sample video:

bash
Copy code
python main.py --video sample_traffic.mp4
Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.

Fork the Repository.

Create a New Branch:

bash
Copy code
git checkout -b feature/your-feature
Commit Your Changes:

bash
Copy code
git commit -am 'Add new feature'
Push to the Branch:

bash
Copy code
git push origin feature/your-feature
Create a Pull Request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
YOLO for the real-time object detection model.
DeepSort for the object tracking algorithm.
