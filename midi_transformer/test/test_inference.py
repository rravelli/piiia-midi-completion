import unittest
import pandas as pd
import os
import pretty_midi
from inference import notes_to_midi


class TestNotesToMidi(unittest.TestCase):
    def setUp(self):
        self.sample_notes_df = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "duration": [0.5, 0.5, 0.5],
                "pitch": [60, 62, 64],
            }
        )

    def test_notes_to_midi_valid(self):
        # Set up output file path
        out_file = "test_output.mid"

        # Test the function
        pm = notes_to_midi(
            self.sample_notes_df, out_file, "Acoustic Grand Piano"
        )

        # Check if the output file is created
        self.assertTrue(os.path.exists(out_file))

        # Validate PrettyMIDI object
        self.assertIsInstance(pm, pretty_midi.PrettyMIDI)
        self.assertEqual(len(pm.instruments), 1)
        self.assertEqual(len(pm.instruments[0].notes), 3)

        # Clean up: remove the test output file
        os.remove(out_file)

    def test_notes_to_midi_invalid_instrument(self):
        # Set up output file path
        out_file = "test_output.mid"

        # Test with an invalid instrument name
        with self.assertRaises(ValueError, msg="Invalid instrument name"):
            notes_to_midi(self.sample_notes_df, out_file, "Invalid Instrument")

    def test_notes_to_midi_invalid_notes_df(self):
        # Set up output file path
        out_file = "test_output.mid"

        # Test with an empty notes DataFrame
        empty_notes_df = pd.DataFrame()
        with self.assertRaises(ValueError, msg="Empty notes DataFrame"):
            notes_to_midi(empty_notes_df, out_file, "Acoustic Grand Piano")

    def test_notes_to_midi_invalid_file_path(self):
        # Test with an invalid output file path
        with self.assertRaises(
            FileNotFoundError, msg="No such file or directory"
        ):
            notes_to_midi(
                self.sample_notes_df,
                "/invalid/path/test_output.mid",
                "Acoustic Grand Piano",
            )

    def test_notes_to_midi_negative_duration(self):
        # Set up output file path
        out_file = "test_output.mid"

        # Test with a note having a negative duration
        self.sample_notes_df.loc[0, "duration"] = -0.5
        with self.assertRaises(
            ValueError, msg="Note duration must be positive"
        ):
            notes_to_midi(
                self.sample_notes_df, out_file, "Acoustic Grand Piano"
            )


if __name__ == "__main__":
    unittest.main()
