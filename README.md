# Covid-19 Warning System

Simple interface for detecting people with/without face masks using YOLOv4 and YOLOv4 Tiny.


## Weights

#### [YOLOv4-tiny]
#### YOLOv4 (in progress)



## Prerequisites

 1. python 3.6 >= and < 3.9
 2. pip


## Usage

0. Note: It is highly recommended to use a GPU because YOLO is a very big network that runs poorly on CPU.
Check [GPU support | Tensorflow] for GPU support.



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

4. Run the detector interface:
    ```shell
    python detector.py
    ```

## Interface

After running detector\.py enter the path for the weights directory you wish to use (dowload weights from weights section):

1. Choose options 1, 2, and 3 for detecting from image, video or cam.
2. Choose options 4 to 9 to change the configurations of the detector (See next section for explaination about the configurations)
3. Choose 10 to see the values of the current configuration.
4. Choose 11 to exit.


## Configurations

1. show masks: set true to highlight masks.
2. show FPS: set true to show video FPS.
3. show scores: set true to show the score of each detection.
4. write detection: set true to save image/video after detecting.
5. score threshold: change score threshold. 
6. IOU threshold: set IOU threshold for non maximum suppression (better leave the default)

## TODO

* [ ] Train YOLOv4 and add weights.
* [ ] Build a simple GUI.
* [ ] Social distancing violation detection.

## References
#### [YOLOv4 paper]
#### [YOLOv4 Source Code]
#### [YOLOv3 paper]
#### [tensorflow-yolov4-tflite] for converting darknet wieghts to tensorflow pb.

Datasets:
* #### [Face Mask Dataset (YOLO Format)]
* #### [Face Mask Detection Dataset]



<!-- Links -->
[YOLOv4 paper]: <https://arxiv.org/abs/2004.10934>
[YOLOv4 Source Code]: <https://github.com/AlexeyAB/darknet>
[YOLOv3 paper]: <https://arxiv.org/abs/1804.02767>
[tensorflow-yolov4-tflite]: <https://github.com/hunglc007/tensorflow-yolov4-tflite>
[GPU support | Tensorflow]: <https://www.tensorflow.org/install/gpu>
[Face Mask Dataset (YOLO Format)]: <https://www.kaggle.com/aditya276/face-mask-dataset-yolo-format>
[Face Mask Detection Dataset]: <https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset>
[YOLOv4-tiny]: <https://drive.google.com/uc?export=download&id=1Rw5CCOxRK52-nFLWKKBAxvKeniJxOr7z>
