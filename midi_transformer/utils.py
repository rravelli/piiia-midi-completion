import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import pretty_midi
from scipy.io import wavfile


def midi_to_wav(midi_file, output_name):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    audio_data = midi_data.synthesize(44100)
    wavfile.write(output_name + ".wav", 44100, audio_data)


def print_accuracy_and_loss(history_path=""):
    with open(history_path + "/history.pkl", "rb") as file:
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

    date = datetime.today().date()
    date = date.strftime("%d-%m-%Y")
    name = f"Accuracy_and_Loss_({date})"
    files = [
        fichier
        for fichier in os.listdir(os.getcwd())
        if os.path.isfile(os.path.join(os.getcwd(), fichier))
    ]
    if name + ".png" in files:
        i = 1
        while name + f"_({i}).png" in files:
            i += 1
        name = name + f"_({i}).png"
    plt.savefig(name)
