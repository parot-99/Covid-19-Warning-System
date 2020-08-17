import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from preprocessing.dataaugmenter import DataAugmenter
from nets.minivggnet import MiniVGGNet
import config


def main():
    try:
        os.mkdir(config.SAVE_TO_FILE)
    except FileExistsError:
        raise FileExistsError('Directory already exits. Try another name.')

    # Load and augment data
    augmenter = DataAugmenter(
        config.DATA_PATH,
        target_size=(32, 32),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
    )

    train_generator, validation_generator = augmenter.flow_from_directory()

    # Initialize keras callbacks to log results and save best model
    logger = CSVLogger(
        os.path.join(config.SAVE_TO_FILE, 'log.csv'),
        append=True
    )

    checkpoint = ModelCheckpoint(
        os.path.join(config.SAVE_TO_FILE, 'net.hdf5'),
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    model = MiniVGGNet.build((32, 32, 3), 1)

    opt = Adam(lr=config.LEARNING_RATE)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        verbose=1,
        callbacks=[logger, checkpoint]
    )

    # Plot loss and accuracy and save results to png file
    history = pd.read_csv(os.path.join(config.SAVE_TO_FILE, 'log.csv'))
    plt.style.use('ggplot')
    figure = plt.figure()
    axes = figure.add_axes([0, 0, 1, 1])
    axes.plot(
        np.arange(0, config.TOTAL_EPOCHS),
        history['loss'],
        label='train_loss'
    )
    axes.plot(
        np.arange(0, config.TOTAL_EPOCHS),
        history['val_loss'],
        label='val_loss'
    )
    axes.plot(
        np.arange(0, config.TOTAL_EPOCHS),
        history['accuracy'],
        label='train_acc'
    )
    axes.plot(
        np.arange(0, config.TOTAL_EPOCHS),
        history['val_accuracy'],
        label='val_acc'
    )
    axes.set_title('Training Loss and Accuracy')
    axes.set_xlabel('Epoch #')
    axes.set_ylabel('Loss/Accuracy')
    axes.legend()
    figure.savefig(config.SAVE_TO_FILE + '/plot.png',
                   bbox_inches='tight',
                   pad_inches=0.5,
                   dpi=200)

    # save info about training to json file
    with open(os.path.join(config.SAVE_TO_FILE, 'info.json'), 'w') as file:
            json.dump(config.INFO, file)


if __name__ == "__main__":
    main()