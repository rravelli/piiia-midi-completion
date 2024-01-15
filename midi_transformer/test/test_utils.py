import unittest
import os
from utils import midi_to_wav, print_accuracy_and_loss
import matplotlib.pyplot as plt
import matplotlib.testing.decorators as mptd

class TestUtils(unittest.TestCase):

    def test_midi_to_wav(self):
        # Test the conversion of a MIDI file to WAV
        midi_file = "/home/momo/piiatransf/piiia-midi-completion/midi_transformer/test/fixture/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--1.midi"
        output_name = "test_output"
        midi_to_wav(midi_file, output_name)

        # Check if the WAV file is created
        wav_file_path = output_name + ".wav"
        self.assertTrue(os.path.exists(wav_file_path))

        # Clean up: remove the test output file
        os.remove(wav_file_path)
    
    @mptd.image_comparison(baseline_images=['accuracy_and_loss_plots'], extensions=['png'])
    def test_print_accuracy_and_loss(self):
        # Test the printing of accuracy and loss graphs
        history_path = "/home/momo/piiatransf/piiia-midi-completion/midi_transformer/test/fixture/"

        # Call the function and generate the plots
        print_accuracy_and_loss(history_path)

if __name__ == '__main__':
    unittest.main()