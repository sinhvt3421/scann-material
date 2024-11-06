import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import random
import time

import numpy as np
import tensorflow as tf
import yaml

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

    config = yaml.safe_load(open(args.dataset))
    print("Create model use Ring Information: ", args.use_ring, "\n")

    config["model"]["feature"] = args.feature
    config["model"]["use_ring"] = args.use_ring
    config["model"]["use_drop"] = args.use_drop

    config["hyper"]["use_ref"] = args.use_ref
    config["hyper"]["target"] = args.target
    config["hyper"]["pretrained"] = args.pretrained

    scann = SCANN(config, args.pretrained)

    print("Load data for dataset: ", args.dataset, " with target: ", args.target, "\n")
    scann.prepare_dataset()

    if args.mode == "train":
        print("Start Model training", "\n")
        start = time.time()
        scann.train(500)

        print("Training time: ", time.time() - start, "\n")

    print("Start Model evaluation:")
    # Evaluate for testset
    scann.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument("target", type=str, help="Target energy for training")

    parser.add_argument("dataset", type=str, help="Path to dataset configs")

    parser.add_argument(
        "--use_ring",
        type=bool,
        default=False,
        help="Whether to use ring/aromatic as extra emedding",
    )

    parser.add_argument(
        "--use_ref",
        type=bool,
        default=False,
        help="Whether to use ref optimization energy",
    )

    parser.add_argument(
        "--use_drop",
        type=bool,
        default=False,
        help="Whether to use dropout in training model",
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="atomic",
        help="Whether to use atomic or cgcnn feature",
    )

    parser.add_argument("--pretrained", type=str, default="", help="Path to pretrained model (optional)")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Whether to train new model or just run the evaluation on pretrained model",
    )

    args = parser.parse_args()
    main(args)
