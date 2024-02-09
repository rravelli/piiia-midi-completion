from unittest import TestCase
import pathlib

from tokenizer import BATCH_SIZE, make_midi_batchesv2, TOKENIZER
from dataset import download_maestro_dataset
from miditok import REMI, TokenizerConfig
from tensorflow import TensorShape
import os


class TestTokenizer(TestCase):
    def test_make_midi_batchesv2(self):
        train_examples, _, _ = download_maestro_dataset()
        train_batches = make_midi_batchesv2(train_examples)
        for (context, input), target in train_batches.take(300):
            break
        self.assertEqual(context.shape, TensorShape([BATCH_SIZE, BATCH_SIZE]))
        self.assertEqual(input.shape, TensorShape([BATCH_SIZE, BATCH_SIZE - 1]))
        self.assertEqual(target.shape, TensorShape([BATCH_SIZE, BATCH_SIZE - 1]))
