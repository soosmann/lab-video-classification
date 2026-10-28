"""
Train our LSTM on extracted features.
"""
import os.path
import time

from data import DataSet
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from models import ResearchModels


def train(data_type: str, seq_length: int, model: str, saved_model: str = None,
          class_limit: int = None, image_shape: tuple = None,
          load_to_memory: bool = False, batch_size: int = 32, nb_epoch: int = 100):
    """
    The training funtion for the LSTM network.

    Parameters:
        data_type (`str`): The type of data to train on ("features" or "images")
        seq_length (`int`): The minimum length of each video sequence to be used
        model (`str`): The model type to use (lstm), used as model name prefix
        saved_model (`str`): The path to a saved Keras model to load if wanted to continue training
        class_limit (`ìnt`): The number of classes to limit to (uses all if None)
        image_shape (`tuple`): The shape of the input images (if using images)
        load_to_memory (`bool`): Whether to load all data into memory or not (decides on which generator to use)
        batch_size (`int`): The batch size to use
        nb_epoch (`int`): The number of epochs to train for

    Returns:
        None
    """
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + str(class_limit) + '_' + str(seq_length) + "-" + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.keras'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + str(class_limit) + '_' + str(seq_length) + "-" + 'training-' + \
        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    print(data.classes)

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit = 5  # int, can be 1-101 or None
    #class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 8
    nb_epoch = 100

    data_type = 'features'
    image_shape = None
    

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
