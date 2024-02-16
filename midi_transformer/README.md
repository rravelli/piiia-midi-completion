# Midi transformer

## Installations

First of all go to the root directory

```bash
cd midi_transformer
```

Then install all dependencies
```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

## Generate midi

Make sure you have training datas copied or generated in the root.
Your repo should look like this:

```
midi_transformer/
    ...
    training_x/
        checkpoint
        cp.ckpt.data...
        cp.ckpt.index
    ...
```

Then generate midi with

```bash
python inference.py
```