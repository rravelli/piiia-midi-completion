import pretty_midi
from scipy.io import wavfile
import glob
import pathlib
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


def midi_to_wav(midi_file, output_name):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    audio_data = midi_data.synthesize(44100)
    wavfile.write(output_name + ".wav", 44100, audio_data)


# data_dir = pathlib.Path("data/maestro-v2.0.0")
# filenames = glob.glob(str(data_dir / "**/*.mid*"))
# print(len(filenames))
# midi_sample = filenames[1000]

# output = "music_test"

# midi_to_wav(midi_sample, output)


def print_accuracy_and_loss(history_path=""):
    with open(history_path + "history.pkl", "rb") as file:
        loaded_history = pickle.load(file)

    plt.figure(figsize=(12, 6))
    # Tracer l'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(loaded_history["masked_accuracy"])
    plt.title("Masked accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks([i for i in range(len(loaded_history["masked_accuracy"]))])

    # Tracer la perte
    plt.subplot(1, 2, 2)
    plt.plot(loaded_history["loss"])
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks([i for i in range(len(loaded_history["loss"]))])
    plt.show()
