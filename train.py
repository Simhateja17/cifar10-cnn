"""Train a three-block CNN on CIFAR-10 and track train vs validation loss and accuracy.

Usage:
    python train.py                 # with dropout
    python train.py --no-dropout    # same model, dropout layers removed
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential

SEED = 42
RESULTS_DIR = "results"
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# Fixed test images (one per 1000, so each run shows the same pictures).
PICKS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]


def load_cifar10():
    # CIFAR10_NPZ can point to a local copy (keys: train_images, train_labels,
    # test_images, test_labels) when the official download is slow.
    path = os.environ.get("CIFAR10_NPZ")
    if path:
        data = np.load(path)
        return (data["train_images"], data["train_labels"]), (data["test_images"], data["test_labels"])
    from keras.datasets import cifar10

    return cifar10.load_data()


def build_model(use_dropout):
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    # Three blocks: two convolutions, then pooling and (optionally) dropout.
    # The first conv of each block keeps the size ('same'), the second shrinks it.
    for filters in (32, 64, 64):
        model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(filters, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if use_dropout:
            model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dropout", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    use_dropout = not args.no_dropout
    name = "dropout" if use_dropout else "no_dropout"

    utils.set_random_seed(SEED)

    (train_images, train_labels), (test_images, test_labels) = load_cifar10()

    # Pixels run 0 to 255; scale to 0 to 1.
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    # Shuffle before validation_split, which takes the LAST 10 percent of the data.
    # Images and labels must use the same order.
    order = np.random.permutation(len(train_images))
    train_images = train_images[order]
    train_labels = train_labels[order]

    train_labels_onehot = utils.to_categorical(train_labels, 10)
    test_labels_onehot = utils.to_categorical(test_labels, 10)

    model = build_model(use_dropout)
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

    history = model.fit(
        train_images,
        train_labels_onehot,
        validation_split=0.1,
        batch_size=32,
        epochs=args.epochs,
    )
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_onehot)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss ({name})")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_loss.png"), dpi=120)

    plt.figure()
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy ({name})")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_accuracy.png"), dpi=120)

    # Predict a few fixed test images and show predicted vs true class.
    probabilities = model.predict(test_images[PICKS])
    plt.figure(figsize=(12, 5.5))
    correct = 0
    for i, index in enumerate(PICKS):
        predicted = CLASS_NAMES[int(np.argmax(probabilities[i]))]
        true = CLASS_NAMES[int(test_labels[index][0])]
        correct += predicted == true
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[index])
        plt.title(f"pred: {predicted}\ntrue: {true}", fontsize=9, color="green" if predicted == true else "red")
        plt.axis("off")
    plt.suptitle(f"Predictions ({name}): {correct} of {len(PICKS)} correct")
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_predictions.png"), dpi=120)

    summary = {
        "run": name,
        "parameters": model.count_params(),
        "epochs": args.epochs,
        "final_train_accuracy": history.history["accuracy"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1],
        "test_accuracy": test_accuracy,
        "final_train_loss": history.history["loss"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "best_val_loss": min(history.history["val_loss"]),
        "best_val_loss_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
