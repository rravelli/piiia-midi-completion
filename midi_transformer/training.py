import datetime
import json
import pickle
from os import makedirs

import keras
import tensorflow as tf
import tensorflow_text as text  # noqa
from dataset import download_maestro_dataset
from model import create_model
from tokenizer import BATCH_SIZE, SEQ_LENGTH, make_midi_batchesv2
from utils import print_accuracy_and_loss

EPOCHS = 1
TRAIN_FILES = 2
VALIDATION_FILES = 2

if __name__ == "__main__":
    # download dataset
    train_examples, test_examples, val_examples = download_maestro_dataset()
    # make batches
    train_batches = make_midi_batchesv2(train_examples, TRAIN_FILES)
    val_batches = make_midi_batchesv2(val_examples, VALIDATION_FILES)

    transformer = create_model()

    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint_path = f"training_data/{date}"
    # Write params
    makedirs(checkpoint_path)
    with open(checkpoint_path + "/params.json", "w") as f:
        params = {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_files": TRAIN_FILES,
            "validation_files": VALIDATION_FILES,
            "seq_length": SEQ_LENGTH,
            "loss": transformer.loss.__name__,
        }
        json.dump(params, f)
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + "/cp.ckpt",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    history = keras.callbacks.History()

    # lancement du training
    history = transformer.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=[cp_callback, history],
    )
    # sauvegarde du model
    tf.saved_model.save(transformer, "saved_model/midi_transformer")
    # sauvegarde de history
    with open("history.pkl", "wb") as file:
        pickle.dump(history.history, file)

    print_accuracy_and_loss(output_dir=checkpoint_path)
