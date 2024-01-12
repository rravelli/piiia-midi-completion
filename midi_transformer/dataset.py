import tensorflow_datasets as tfds

def download_dataset(): 
    examples, metadata = tfds.load(
        "ted_hrlr_translate/pt_to_en", with_info=True, as_supervised=True
    )

    return examples["train"], examples["validation"]
