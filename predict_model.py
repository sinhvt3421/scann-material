import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle

import numpy as np
import tensorflow as tf
import yaml
import random

from sklearn.metrics import mean_absolute_error, r2_score

from scann.models import SCANN


def set_seed(seed=2134):
    # tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(args):
    set_seed(0)
    config = yaml.safe_load(open(os.path.join(args.trained_model, "config.yaml")))

    print("Load pretrained weight for target ", config["hyper"]["target"])
    scann = SCANN(
        config,
        os.path.join(
            args.trained_model,
            "models",
            "model_{}.h5".format(config["hyper"]["target"]),
        ),
        mode="infer",
    )

    print("Load data for trained model: ", config["hyper"]["data_energy_path"])
    scann.prepare_dataset(split=False)

    ga_scores = []
    struct_energy = []
    y = []
    idx = 0
    data = scann.dataIter

    for i in range(len(data)):
        inputs, target = data.__getitem__(i)
        energy, attn_global = scann.predict_data(inputs)

        ga_scores.extend(attn_global)

        struct_energy.extend(list(np.squeeze(energy)))
        y.extend(list(target))

        idx += data.batch_size
        if i % 10 == 0:
            print(idx)

    print(r2_score(struct_energy, y), mean_absolute_error(struct_energy, y))

    print("Save prediction and GA score")
    pickle.dump(
        ga_scores,
        open(
            os.path.join(
                args.trained_model,
                "ga_scores_{}.pickle".format(config["hyper"]["target"]),
            ),
            "wb",
        ),
    )

    pickle.dump(
        [y, struct_energy],
        open(
            os.path.join(
                args.trained_model,
                "energy_pre_{}.pickle".format(config["hyper"]["target"]),
            ),
            "wb",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("trained_model", type=str, help="Target trained model path for loading")

    args = parser.parse_args()
    main(args)
