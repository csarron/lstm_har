import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))


# modified from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

def freeze_graph(model_folder, frozen_model_name):
    # We retrieve our checkpoint full path
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our frozen graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/{}.pb".format(frozen_model_name)

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["output", "accuracy", "cost"]
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    print("begin loading model")

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        print("model loaded")
        # export network weights and biases to text files
        weights = ["w_in", "b_in", "w_out", "b_out",
                   "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights",
                   "rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases",
                   "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights",
                   "rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases"]
        for name in weights:
            v = sess.run("{}:0".format(name))
            var_file_name = "data/{}.csv".format(name.replace("/", "_"))
            print("save {} to file: {}".format(name, var_file_name))
            np.savetxt(var_file_name, v, delimiter=",")

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            output_node_names=output_node_names  # The output node names are used to select the useful nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        with tf.gfile.FastGFile("{}.txt".format(output_graph), "wb") as f:
            f.write(str(output_graph_def))

        print("%d ops in the final graph." % len(output_graph_def.node))


def do_freeze(model_folder, frozen_model_name):
    freeze_graph(model_folder, frozen_model_name)

    print("frozen graph bin saved to: {}.pb".format(frozen_model_name))
    print("frozen graph text saved to: {}.pb.txt".format(frozen_model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",
                        type=str, default="data", help="model folder(contains *.model.meta, *.model.data-* etc")
    parser.add_argument("--frozen_model_name",
                        type=str, default="lstm_model", help="frozen model name")
    args = parser.parse_args()

    do_freeze(args.model_folder, args.frozen_model_name)
