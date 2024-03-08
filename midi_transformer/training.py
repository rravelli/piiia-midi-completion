import datetime
import pickle

import keras
import tensorflow as tf
import tensorflow_text as text  # noqa
from dataset import download_maestro_dataset
from model import create_model
from tokenizer import make_midi_batchesv2

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    # download dataset
    train_examples, test_examples, val_examples = download_maestro_dataset()
    # make batches
    train_batches = make_midi_batchesv2(train_examples, 2)
    val_batches = make_midi_batchesv2(val_examples, 2)

    transformer = create_model()

    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint_path = f"training_data/{date}/cp.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path + "/cp.ckpt",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    history = keras.callbacks.History()

    # lancement du training
    transformer.fit(
        train_batches,
        epochs=3,
        validation_data=val_batches,
        callbacks=[cp_callback, history],
    )
    # sauvegarde du model
    tf.saved_model.save(transformer, "saved_model/midi_transformer")
    # sauvegarde de history
    with open("history.pkl", "wb") as file:
        pickle.dump(history.history, file)
