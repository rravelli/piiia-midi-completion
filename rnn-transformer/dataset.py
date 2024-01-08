import tensorflow_datasets as tfds
from tokenizer import make_batches

examples, metadata = tfds.load(
    "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
)

train_examples, val_examples = examples["train"], examples["validation"]

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)
