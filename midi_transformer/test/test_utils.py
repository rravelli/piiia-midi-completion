import unittest
import os
from utils import midi_to_wav, print_accuracy_and_loss

class TestUtils(unittest.TestCase):

    def test_midi_to_wav(self):
        # Test the conversion of a MIDI file to WAV
        midi_file = "/home/momo/piiatransf/piiia-midi-completion/midi_transformer/test/fixture/MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--1.midi"  # Replace with an actual MIDI file path
        output_name = "test_output"
        midi_to_wav(midi_file, output_name)

        # Check if the WAV file is created
        wav_file_path = output_name + ".wav"
        self.assertTrue(os.path.exists(wav_file_path))

        # Clean up: remove the test output file
        os.remove(wav_file_path)

if __name__ == '__main__':
    unittest.main()