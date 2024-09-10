Certainly! Here's a detailed `README.md` for your Traffic-YOLO project:

---

# Traffic-YOLO

**Traffic-YOLO** is an advanced real-time object detection and tracking system designed to improve traffic management and analysis. By combining YOLO (You Only Look Once) for efficient object detection with DeepSort for robust tracking, this project offers a powerful tool for monitoring traffic scenarios.

## Features

- **Real-time Object Detection:** Detects and identifies traffic-related objects such as vehicles and pedestrians in real-time using YOLO.
- **Object Tracking:** Maintains consistent tracking of detected objects across video frames with DeepSort.
- **Distance Measurement:** Displays the distance to detected objects on the video frame for enhanced situational awareness.
- **Leader Lines:** Visualizes leader lines pointing towards destinations while avoiding obstacles, aiding in navigation and traffic flow analysis.

## Installation

To get started with Traffic-YOLO, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/joelprince2601/Traffic-YOLO.git
   cd Traffic-YOLO
   ```

2. **Install Dependencies:**

   Make sure you have Python 3.7 or higher installed. Then, install the required Python packages using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO Weights:**

   Download the YOLO weights file from [YOLO's official website](https://pjreddie.com/darknet/yolo/) and place it in the project directory.

4. **Download DeepSort Model:**

   Obtain the DeepSort model weights from [DeepSort's repository](https://github.com/nwojke/deep_sort).

## Usage

1. **Run the Detection and Tracking Script:**

   Execute the following command to start the object detection and tracking system on your video feed:

   ```bash
   python main.py --video <path_to_video_file>
   ```

   Replace `<path_to_video_file>` with the path to your video file or use `--source 0` to use your webcam.

2. **Adjust Configuration:**

   Modify the `config.py` file to adjust settings such as detection thresholds, model paths, and other parameters.

## Example

Here is an example command to run the script with a sample video:

```bash
python main.py --video sample_traffic.mp4
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional features, please open an issue or submit a pull request.

1. **Fork the Repository.**
2. **Create a New Branch:**

   ```bash
   git checkout -b feature/your-feature
   ```

3. **Commit Your Changes:**

   ```bash
   git commit -am 'Add new feature'
   ```

4. **Push to the Branch:**

   ```bash
   git push origin feature/your-feature
   ```

5. **Create a Pull Request.**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLO](https://pjreddie.com/darknet/yolo/) for the real-time object detection model.
- [DeepSort](https://github.com/nwojke/deep_sort) for the object tracking algorithm.

---

