import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference as opt_inference
from tensorflow.python.training import saver as saver_lib


# modified from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
def freeze_graph(layer, unit, input_names, output_names, accuracy):

    frozen_model_path = "data/{}layer{}unit.pb".format(layer, unit)
    checkpoint_file = "data/{}layer{}unit.ckpt".format(layer, unit)
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
        # export network weights and biases to text files
        if not __debug__:
            output_nodes = "w_in,b_in,w_out,b_out"
            rnn_nodes = [",rnn/multi_rnn_cell/cell_{}/basic_lstm_cell/weights," \
                         "rnn/multi_rnn_cell/cell_{}/basic_lstm_cell/biases".format(i, i) for i in range(args.layer)]
            weights = output_nodes + "".join(rnn_nodes)
            for name in weights.split(","):
                v = sess.run("{}:0".format(name))
                var_file_name = "data/{}.csv".format(name.replace("/", "_"))
                print("save {} to file: {}".format(name, var_file_name))
                np.savetxt(var_file_name, v, delimiter=",")

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
        with tf.gfile.GFile(frozen_model_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("frozen graph binary saved to: {}".format(frozen_model_path))

        frozen_model_text_path = "data/{}layer{}unit{}.pb.txt".format(layer, unit, accuracy)
        with tf.gfile.FastGFile(frozen_model_text_path, "wb") as f:
            f.write(str(output_graph_def))
            print("frozen graph text saved to: {}".format(frozen_model_text_path))

        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=2,
                        help="lay size of the LSTM model")
    parser.add_argument("--unit", type=int, default=64,
                        help="hidden unit of the LSTM model")
    parser.add_argument("--input_names", type=str, default="input",
                        help="Input node names, comma separated")
    parser.add_argument("--output_names", type=str, default="output",
                        help="The name of the output nodes, comma separated")
    parser.add_argument("--accuracy", type=str, default="",
                        help="accuracy of the LSTM model")
    args = parser.parse_args()
    freeze_graph(args.layer, args.unit, args.input_names, args.output_names, args.accuracy)
