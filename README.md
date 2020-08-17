In the first section are commands to be used directly. The rest of the file
gives more detailed information about how to use each module. 

ps: Navigate to the directory containing this file (README.txt) using bash,
    cmd or any CLI before running any commands.

ps: Before running any commands, all dependencies must be isntalled, see
    section 1.1 

ps: To use gpu for training and testing please refer to tensorflow guide for
    running tesnorflow on gpu.

ps: All packages and thier versions are in the requirements.txt file


0) Quick guide

    Test ResNet: python test_net.py -d dataset/test

    Test DenseNet: python test_net.py -d dataset/test/ -m trained_models/net.
    hdf5 -r False

    Test MiniVGG: python test_net.py -d dataset/test -m trained_nets/MiniVGG/
    net.hdf5 -r false -s 32

    Test ResNet on 1 image and show the image:
            python test_net.py -i path/to/image.jpg


1) How to use:

    The project contains 3 pre-trained networks and the results obtained.

    The following instructions can be used to test the already trained models
    or train a new model using different configuration.

    1.1- Building the environment:

        To install dependencies the machine must have python 3.7>= and pip
        instaleld.

        The dependencies can be isntalled directly or a better soltion is to
        use a python virtual environment.

        (optional) {

            The following instructions are for creating a new
            environment:

                -Install virtualenv: python -m pip install virtualenv

                -Create new environment: pythom -m venv environment-path

            To activate the environment use:

                POSIX   bash/zsh            $ source <venv>/bin/activate
                        fish                $ . <venv>/bin/activate.fish
                        csh/tcsh            $ source <venv>/bin/activate.csh
                        PowerShell Core     $ <venv>/bin/Activate.ps1

                Windows cmd.exe             C:\> <venv>\Scripts\activate.bat
                        PowerShell          PS C:\> <venv>\Scripts\Activate.ps1
        }

        To isntall the requirements use the following command, if a virtual
        environment is used and activated, the requirements will be isntalled
        in that virtual environment:

            -Install requirments:  pip install -r requirments.txt

               
    1.2- Testing the models:

         Test a model using test_net.py module. The module requires multiple
         flags:

            -d --data: test data path
            -i --image: test image path
            -m --model: model path. Default=Resnet
            -r --resnet: Set False when not using ResNet. Default=True
            -s --size: input size of the network used. Default=224

        ResNet needs special  preprocessing before using, because of that a 
        ResNet flag is used to imply that ResNet is used.

        1.2.1-  Using a test set:

                The directory containing the test data must have the following
                structure:

                    |__ test
                        |______ mask: [0.jpg, 1.jpg, 2.jpg ....]
                        |______ no-mask: [0.jpg, 1.jpg, 2.jpg ...]

                To test the data run the following command:

                    python -d test_directory_path -m model_path -s input_size
                    -r True(if ResNet is used else False)

                    ex: python -d dataset/test -m trained_nets/DenseNet/net.
                        hdf5

        1.2.2-  Using one image:

                To test a model on one image run the following command:

                    python -i image_path -m model_path -s input_size -r True/
                    False
         
                    ex: python -i path/to/image.jpg -m trained_nets/DenseNet/
                        net.hdf5 

2) Contents:

    Head of line information about the contents. For more information about
    how a specific package or module work please refer to the module or
    package documentation.

    2.1-    Dataset -> Directory:
            Directory used to store the dataset used for training and valdiation data.
            The directory must have the following structure:

                |__ train
                    |______ mask: [0.jpg, 1.jpg, 2.jpg ....]
                    |______ no-mask: [0.jpg, 1.jpg, 2.jpg ...]
                |__ validation
                    |______ mask: [0.jpg, 1.jpg, 2.jpg ....]
                    |______ no-mask: [0.jpg, 1.jpg, 2.jpg ...]

    2.2-    dl-env -> Directory:
            Directory containing the python virtual environment that contains
            all the dependencies required to recreate the project.

    2.3-    nets -> python package:
            Package containing the neural networks that will be used.

        3.1-    minivggbet.py -> module:
                A VGG-like network architecture.

        3.2-    nethead.py -> module:
                The fully connected head to replace DenseNet and ResNet
                original head network.

    2.4-    preprocessing -> python package:
            Package containing necessary modules to load, augment, and
            preprocess data.

        4.1-    aspectwarepreprocessor.py -> module:
                A Preprocessor that resizes images respecting the aspect ware
                ratio.

        4.2-    dataaugmenter.py -> module:
                Module to augment data using keras functionalties and flow the
                data from directory.

        4.3-    datasetloader.py -> module:
                Used to load images into memory. Efficient when testing as
                small set of images.
                Can use any preprocessor that returns a numpy array containing
                new image preprocessed while loading data.

        4.4-    imagetoarray.py -> module:
                Simple preprocessor that uses keras image_to_array method.

        4.5-    resnetpreprocessor.py -> module:
                Preprocessor to normalize pixels betwwen -1 and 1 to be used
                in ResNet50

    2.5-    trained_nets -> Directory:
            Used to save the trained networks wieghts, the results
            obtained from training, and information about how the training
            was done to recreate the training processes.

    2.6-    config.py -> module:
            Used to change the training process configuration.

    2.7-    test_network.py -> module:
            Module to load a network (using hdf5 format) and test the network
            on a test set or 1 image and show the image using open-cv.

    2.8-    train.py -> module:
            Used for training networks.

    2.9-    requirments.txt -> text file:
            Contains the dependencies and thier versions.