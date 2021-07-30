"""Convert Huggingface T5 Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
import torch

from transformers import T5Model


def convert_t5_pytorch_checkpoint_to_tf(model: T5Model, ckpt_dir: str, model_name: str, variables_map: dict):

    """
    Args:
        model: BertModel Pytorch model instance to be converted
        ckpt_dir: Tensorflow model directory
        model_name: model name
        variable_map: dictionary with two keys:
            "variable_map": mapping between pytorch and tf variable names "transposed_tensors" tensors that have been
            transposed from tf
    """

    tensors_to_transpose = variables_map["transposed_tensors"]

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    var_map = variables_map["variables_map"]

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.compat.v1.get_variable(
            dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer()
        )
        session.run(tf.compat.v1.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    bad = []
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for var_name in state_dict:
            tf_name = var_map.get(var_name) or var_name
            if var_name not in var_map:
                bad.append(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print(f"Successfully created {tf_name}: {np.allclose(tf_weight, torch_tensor)}")

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))
        for e in bad:
            print(e)


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="name the output model")
    parser.add_argument(
        "--pytorch_path", type=str, required=True, help="path to the t5 pytorch checkpoint (with config file)"
    )
    parser.add_argument("--tf_path", type=str, required=True, help="Directory in which to save tensorflow model")
    parser.add_argument("--variables_map", type=str, required=True, help="Path to the variable name mappings")
    args = parser.parse_args(raw_args)

    model = T5Model.from_pretrained(
        pretrained_model_name_or_path=args.pytorch_path,
    )

    with open(args.variables_map) as fin:
        variables_map = json.load(fin)

    convert_t5_pytorch_checkpoint_to_tf(
        model=model, ckpt_dir=args.tf_path, model_name=args.model_name, variables_map=variables_map
    )


if __name__ == "__main__":
    main()
