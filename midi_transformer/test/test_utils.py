import os
import unittest
from datetime import datetime
import pathlib

from utils import midi_to_wav, print_accuracy_and_loss


class TestUtils(unittest.TestCase):
    def test_midi_to_wav(self):
        # Test the conversion of a MIDI file to WAV
        midi_file = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).joinpath(
            "fixture/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--1.midi"
        )
        midi_file = str(midi_file)
        output_name = "test_output"
        midi_to_wav(midi_file, output_name)

        # Check if the WAV file is created
        wav_file_path = output_name + ".wav"
        self.assertTrue(os.path.exists(wav_file_path))

        # Clean up: remove the test output file
        os.remove(wav_file_path)

    def test_print_accuracy_and_loss(self):
        date = datetime.today().date()
        date = date.strftime("%d-%m-%Y")
        out_file = f"Accuracy_and_Loss_({date}).png"
        # Test the function
        path = str(
            pathlib.Path(os.path.abspath(os.path.dirname(__file__))).joinpath("fixture")
        )
        print_accuracy_and_loss(history_path=path)
        # Check if the output file is created
        self.assertTrue(os.path.exists(out_file))
        # Clean up: remove the test output file
        os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
