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

## Copyright

The MAESTRO dataset used to train our Transformer, was introduced in the following paper :
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}