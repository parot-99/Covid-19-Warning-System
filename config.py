"""Arguments:
    DATA_PATH: String specifying the path to dataset to be used
    The directory must have the following structre:

    LEARNING_RATE: Integer specifying learning rate.

    BATCH_SUZE: Integer specifying batch size.

    EPOCHS: # of epochs.

    TOTAL_EPOCHS: total # of epochs used to fully train the network.

    SAVE_TO_FILE: String specifying the path and new directory name to save
    network information in.

    INFO: Add info about number of epochs used and the optimizer
    associated with the set of epochs using the folloing syntax:
    epochs-1: number of epochs,
    optimizer-1: {'name': ..., 'lr': learning rate, 'decay': ...},
    .
    .
    epochs-n: ...,
    optimizer-n: ...
"""

DATA_PATH = 'dataset'
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 25
SAVE_TO_FILE = 'trained_nets/test'
TOTAL_EPOCHS = 25
INFO = {
    'epochs-1': 25,
    'optimizer-1': {'name': 'Adam', 'lr': 1e-4},
}
