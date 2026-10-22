# CIFAR-10 image classification with a CNN (Keras / TensorFlow)

A three-block convolutional network trained on CIFAR-10, with train and validation loss and accuracy tracked every epoch. The same model is trained twice, with and without dropout, to see how dropout affects overfitting.

## Setup

- Data: CIFAR-10, 50,000 training images and 10,000 test images, 10 classes, 32x32 color.
- Preprocessing: pixels scaled to 0 to 1, labels one-hot encoded, training images shuffled together with their labels before the validation split (Keras `validation_split` takes the last 10 percent, so unshuffled class-ordered data would give a validation set of one class).
- Split: 45,000 images for training, 5,000 for validation, 10,000 for the final test.
- Model: 3 blocks of two 3x3 conv layers (32, 64, 64 filters) + max pooling, then Dense 512 and a 10-way softmax. ReLU activations. 276,138 parameters. Dropout of 0.25 after each block and 0.5 after the Dense layer in the dropout run.
- Training: nadam, categorical cross-entropy, batch size 32, 20 epochs, seed 42.

## Results

One run per row (seed 42). Final epoch values, test accuracy from the 10,000 test images.

| Run | Train acc | Val acc | Test acc | Final train loss | Final val loss | Best val loss |
|---|---|---|---|---|---|---|
| With dropout | 0.767 | 0.757 | 0.772 | 0.687 | 0.706 | 0.698 (epoch 17) |
| No dropout | 0.924 | 0.727 | 0.740 | 0.219 | 1.485 | 0.855 (epoch 3) |

What the runs show:

- Without dropout, training loss keeps falling (to 0.22) while validation loss bottoms out around epoch 3 and then rises to 1.48. The model keeps improving on images it trains on and gets worse on images it has not seen: overfitting. The gap between train and validation accuracy is 0.197.
- With dropout, train and validation loss stay together for the whole run, and the train to validation accuracy gap is 0.010.

What they do not show:

- Each row is a single seeded run. In my earlier unseeded runs of the same model, test accuracy moved by about 0.04 from run to run, so the 0.03 difference in test accuracy here is within that noise. I only claim the change in overfitting behavior, not a better final test accuracy.
- Training accuracy with dropout is measured while dropout is on, which makes it look lower than it would on the same images with dropout off.

Plots and raw numbers are in `results/`:

| | Loss | Accuracy |
|---|---|---|
| With dropout | `results/dropout_loss.png` | `results/dropout_accuracy.png` |
| No dropout | `results/no_dropout_loss.png` | `results/no_dropout_accuracy.png` |

Summary numbers: `results/dropout_summary.json`, `results/no_dropout_summary.json`. Full training logs: `results/*_log.txt`.

## Run it

```
pip install -r requirements.txt
python train.py                # with dropout
python train.py --no-dropout   # same model, no dropout layers
```

The CIFAR-10 download goes through `keras.datasets`. If that is slow, set `CIFAR10_NPZ` to a local `.npz` with the keys `train_images`, `train_labels`, `test_images`, `test_labels`.

Tested with TensorFlow 2.20 and Keras 3.10.
