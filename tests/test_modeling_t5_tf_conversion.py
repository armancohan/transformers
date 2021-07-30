import numpy as np
import tensorflow as tf


def test_conversion(checkpoint1, checkpoint2):
    init_vars1 = dict(tf.train.list_variables(checkpoint1))
    init_vars2 = dict(tf.train.list_variables(checkpoint2))

    # these keys are allowed to be missing from the converted checkpoint
    allowed_missed_keys = [
        "adam_v",
        "adam_m",
        "AdamWeightDecayOptimizer",
        "AdamWeightDecayOptimizer_1",
        "global_step",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def check_variables(vars1, vars2):
        for k, v in vars1.items():
            if k not in vars2 and k not in allowed_missed_keys and "_slot_" not in k:
                assert False
        return True

    assert check_variables(init_vars1, init_vars2)
    assert check_variables(init_vars2, init_vars1)

    def get_weights(checkpoint, variable_names):
        tf_weights = {}
        for name, shape in variable_names.items():
            array = tf.train.load_variable(checkpoint, name)
            tf_weights[name] = array
        return tf_weights

    weights1 = get_weights(checkpoint1, init_vars1)
    weights2 = get_weights(checkpoint2, init_vars2)

    def check_weights(weights1, weights2):
        for k, v in weights1.items():
            if k not in weights2:
                continue  # already checked above
            if not np.allclose(v.astype(np.float32), weights2[k].astype(np.float32)):
                assert False
        return true

    assert check_weights(weights1, weights2)
    assert check_weights(weights2, weights1)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint1", help="")
    parser.add_argument("--checkpoint2", help="")
    args = parser.parse_args()
    test_conversion(args.checkpoint1, args.checkpoint2)


if __name__ == "__main__":
    main()
