import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import argparse
import pickle

import numpy as np
import yaml

from scann.models import SCANN
from scann.utils.general import load_file, prepare_input_pmt


def main(args):
    config = yaml.safe_load(open(os.path.join(args.trained_model, "config.yaml")))

    print("Reading input from file: ", args.file_name)
    struct = load_file(
        args.file_name, mol=True
    )  # True if molecule file else False for crystals, file format support by Pymatgen

    inputs = prepare_input_pmt(struct, d_t=4.0, w_t=0.4, angle=True)  # angle = True if use SCANN+, else False for SCANN

    print("Load pretrained weight for target ", config["hyper"]["target"])
    model = SCANN.load_model_infer(
        os.path.join(args.trained_model, "models", "model_{}.h5".format(config["hyper"]["target"]))
    )

    print("Prediction for input ")

    energy, attn_global = model.predict(inputs)

    print("Save prediction and GA score")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    struct_name = os.path.splitext(os.path.basename(args.file_name))[0]
    save_xyz = "{}_ga_scores_{}.xyz".format(struct_name, config["hyper"]["target"])

    with open(os.path.join(args.save_path, save_xyz), "w") as f:
        f.write(str(len(struct)) + "\n")
        f.write("XXX \n")
        for i in range(len(struct)):
            f.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    struct.sites[i].label,
                    struct.sites[i].x,
                    struct.sites[i].y,
                    struct.sites[i].z,
                    attn_global[0][i][0],
                )
            )

    pickle.dump(
        [inputs, energy, attn_global],
        open(os.path.join(args.save_path, struct_name + "_ga_scores.pickle"), "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("trained_model", type=str, help="Target trained model path for loading")

    parser.add_argument("save_path", type=str, help="Save path for prediction")

    parser.add_argument("file_name", type=str, help="Path to structure data xyz files")

    args = parser.parse_args()
    main(args)
