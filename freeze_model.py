import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference as opt_inference
from tensorflow.python.training import saver as saver_lib


# modified from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
def freeze_graph(checkpoint_file, frozen_model_name, input_names, output_names):
    # We precise the file fullname of our frozen graph
    absolute_model_folder = "/".join(checkpoint_file.split('/')[:-1])
    output_graph = absolute_model_folder + "/{}.pb".format(frozen_model_name)
    if not saver_lib.checkpoint_exists(checkpoint_file):
        print("Checkpoint file '" + checkpoint_file + "' doesn't exist!")
        exit(-1)

    print("begin loading model")
    saver = tf.train.import_meta_graph(checkpoint_file + '.meta', clear_devices=True)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)
        print("model loaded")
        # # export network weights and biases to text files
        # weights = ["w_in", "b_in", "w_out", "b_out",
        #            "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights",
        #            "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases",
        #            "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights",
        #            "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases"]
        # for name in weights:
        #     v = sess.run("{}:0".format(name))
        #     var_file_name = "data/{}.csv".format(name.replace("/", "_"))
        #     print("save {} to file: {}".format(name, var_file_name))
        #     np.savetxt(var_file_name, v, delimiter=",")

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names=output_names.split(",")  # The output node names are used to select the useful nodes
        )

        # optimize graph
        output_graph_def = opt_inference(output_graph_def, input_names.split(","),
                                         output_names.split(","), dtypes.float32.as_datatype_enum)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        with tf.gfile.FastGFile("{}.txt".format(output_graph), "wb") as f:
            f.write(str(output_graph_def))

        print("%d ops in the final graph." % len(output_graph_def.node))


def do_freeze(checkpoint_file, frozen_model_name, input_names, output_node_names):
    freeze_graph(checkpoint_file, frozen_model_name, input_names, output_node_names)
    print("frozen graph bin saved to: {}.pb".format(frozen_model_name))
    print("frozen graph text saved to: {}.pb.txt".format(frozen_model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_file", type=str, default="data/lstm_har.ckpt",
                        help="TensorFlow variables checkpoint file to load.")
    parser.add_argument("--frozen_model_name", type=str, default="lstm_model_mobile",
                        help="frozen model name")
    parser.add_argument("--input_names", type=str, default="input",
                        help="Input node names, comma separated.")
    parser.add_argument("--output_names", type=str, default="output",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()
    do_freeze(args.checkpoint_file, args.frozen_model_name, args.input_names, args.output_names)
