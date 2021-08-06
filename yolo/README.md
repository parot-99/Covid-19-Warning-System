# CLI

## Usage

* View arguments
    ```shell
    python detector.py -h
    ```
* Detect mask from image
     ```shell
    python detector.py detect mask ./image_path.jpg
    ```
* Detect mask from cam
     ```shell
    python detector.py cam mask cam_source
    ```
* Detect mask from video
     ```shell
    python detector.py demo mask ./video_path.mp4
    ```
<!-- * Detect social distancing from video
     ```shell
    python detector.py detect mask ./image_path.jpg
    ``` -->
## Config.json
* weightsPath: path to the model weights
* showMasks: set true to highlight masks in mask detection
* showFPS: set true to log FPS
* showScores: set true to show the score for each bounding box
* writeDetection: set true to save results
* dontShow: set true to hide video while parsing it
* scoreThreshold: score threshold (from 0.0 to 1.0) 
* iouThreshold: IOU threshold (from 0.0 to 1.0)
* videoOutputPath: path to save video results
* imageOutputPath: path to save image results