import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference as opt_inference
from tensorflow.python.training import saver as saver_lib
import data_util


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


def freeze_data(data_size=500, data_filename="phone_data"):
    input_name = "x"
    label_name = "y"
    np.random.seed(0)
    x, y = data_util.get_data("test")
    samples = np.random.randint(0, len(y), data_size)
    print("use {} data samples".format(data_size))
    if not __debug__:
        print("use samples: {}".format(samples))
    x = x[samples]
    y = np.argmax(y[samples], axis=1) + 1
    print("{} shape: {}".format(input_name, x.shape))
    print("{} shape: {}".format(label_name, y.shape))
    print("save {} to text file at: {}".format(input_name, "data/data.{}.txt".format(input_name)))
    print("save {} to text file at: {}".format(label_name, "data/data.{}.txt".format(label_name)))
    np.savetxt("data/data.{}.txt".format(input_name), np.reshape(x, [data_size, np.prod(x.shape)/data_size]), '%.7e')
    np.savetxt("data/data.{}.txt".format(label_name), y, '%d')

    frozen_data_path = "data/data.pb"
    frozen_data_text_path = "data/data.pb.txt"

    input_const = tf.constant(x, dtype=tf.float32, shape=x.shape, name=input_name)
    label_const = tf.constant(y, dtype=tf.int32, shape=y.shape, name=label_name)

    graph = tf.get_default_graph()
    with tf.Session() as sess:
        sess.run(input_const)
        sess.run(label_const)
        with tf.gfile.GFile(frozen_data_path, "wb") as f:
            f.write(graph.as_graph_def().SerializeToString())
            print("frozen {} and {} to binary file at: {}".format(input_name, label_name, frozen_data_path))
        with tf.gfile.FastGFile(frozen_data_text_path, "wb") as f:
            f.write(str(graph.as_graph_def()))
            print("frozen {} and {} to text file at: {}".format(input_name, label_name, frozen_data_text_path))

    with tf.gfile.GFile(frozen_data_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None,
                            name="", op_dict=None, producer_op_list=None)
    session = tf.Session(graph=graph)
    input_op = graph.get_operation_by_name(input_name).outputs[0]
    label_op = graph.get_operation_by_name(label_name).outputs[0]

    input_op_result = session.run(input_op)
    label_op_result = session.run(label_op)

    assert input_op_result.shape == x.shape
    assert label_op_result.shape == y.shape

    assert np.allclose(x, input_op_result)
    assert np.allclose(y, label_op_result)
    data_util.zip_files("model/{}.zip".format(data_filename), "data/data.*")


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
    # freeze_graph(args.layer, args.unit, args.input_names, args.output_names, args.accuracy)
    freeze_data()
