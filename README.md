# Covid-19 Warning System

A deep learning based system that can:  
* Detect people not wearing face masks, mark them, and count the number of violations. 
* Detect social distancing violations, mark them, and count the number of violations.

## Weights
* yolov4-mask (in progress) -> mask detection
* [yolov4-tiny-mask] -> mask detection
* yolov4 (in progress) -> social distancing detection
* yolov4-tiny (in progress) -> social distancing detection

## Prerequisites
 1. Python 3.5–3.8
 2. pip 19.0 or later
   
## Note
    The requirements are updated acording to which version of tensorflow the repository was lately tested on.
    Feel free to use Tensorflow 2.2-2.4


## Installation
1. Note: For best performance, enable GPU for tensorflow. Check [GPU support | Tensorflow] for GPU support.

1. Clone the repository 

2. (Optional) Create a virtual enviroment:
    ```shell
    python -m venv mask-detector-env
    ```
    and activate the enviroment using:

    Platform | Second Header | Command to activate virtual environment
    ------------ | ------------- | -------------
    POSIX | bash/zsh | $ source <venv>/bin/activate
    || fish | $ source <venv>/bin/activate.fish
    || csh/tcsh | $ source <venv>/bin/activate.csh
    || PowerShell Core |  <venv>/bin/Activate.ps1
    Windows | cmd.exe | C:\> <venv>\Scripts\activate.bat
    || PowerShell | PS C:\> <venv>\Scripts\Activate.ps1

3. Install requirments:

    ```shell
    pip install -r requirements.txt
    ```

4. Download weights to be used from weights section
   * yolov4-mask or yolov4-tiny-mask for mask detection
   * yolov4 or yolov4-tiny for social distancing detection 

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
    showMasks: set true to highlight masks in mask detection
    showFPS: set true to log FPS
    showScores": set true to show the score for each bounding box
    writeDetection": set true to save results
    scoreThreshold": score threshold (from 0.0 two 1.0)
    iouThreshold": IOU threshold (from 0.0 two 1.0)
    detectorPath": path to the detector to be used 
    videoOutputPath": path to save video results
    imageOutputPath": path to save image results

## TODO

* [ ] Train yolov4-mask and add weights
* [ ] Social distancing detection
* [ ] Add yolov4 weights
* [ ] Add yolov4-tiny weights
* [ ] Build a simple GUI

## Dataset
    We collected images from multiple sources (mentioned is the Acknowledgements section), removed images that may lead to an unwanted results, added new images for better results, and labeled all the images.

    The new [mask-dataset] used to train the face mask detector.

## Tech Stack
* Python
* tensorflow
* Opencv
* Numpy
* YOLO (Darknet framework)

## References
* [YOLOv4 paper]
* [YOLOv4 Source Code]
* [YOLOv3 paper]
* [tensorflow-yolov4-tflite] for converting darknet wieghts to tensorflow pb.

## Acknowledgements:
* Dataset
  * [Face Mask Dataset (YOLO Format)]
  * [Face Mask Detection Dataset]
  * [WIDER FACE]
  * [Google] Search Engine

## Authors
* [Ahmed Abdulkader]
* [M.Farok Mohammed]

# License
[BSD 3-Clause License]

<!-- Links -->
[YOLOv4 paper]: <https://arxiv.org/abs/2004.10934>
[YOLOv4 Source Code]: <https://github.com/AlexeyAB/darknet>
[YOLOv3 paper]: <https://arxiv.org/abs/1804.02767>
[tensorflow-yolov4-tflite]: <https://github.com/hunglc007/tensorflow-yolov4-tflite>
[GPU support | Tensorflow]: <https://www.tensorflow.org/install/gpu>
[Face Mask Dataset (YOLO Format)]: <https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format>
[Face Mask Detection Dataset]: <https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset>
[WIDER FACE]: <http://shuoyang1213.me/WIDERFACE/>
[Google]: <https://www.google.com/>
[yolov4-mask]: <>
[yolov4-tiny-mask]: <https://drive.google.com/uc?export=download&id=1Rw5CCOxRK52-nFLWKKBAxvKeniJxOr7z>
[yolov4]: <>
[yolov4-tiny]: <>
[BSD 3-Clause License]: <https://github.com/parot-99/Covid-19-Warning-System/blob/master/LICENSE>
[mask-dataset]: <https://drive.google.com/uc?export=download&id=1z4xdhhTcGHx3bDmbdc2dc-MffZKZSNdd>
[Ahmed Abdulkader]: <https://github.com/parot-99>
[M.Farok Mohammed]: <https://github.com/farok-amo>
