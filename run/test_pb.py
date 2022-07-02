from tensorflow.core.framework import node_def_pb2
import re
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
from tensorflow.python.platform import gfile
from collections import defaultdict
import numpy as np




def get_selected_layer(graph_def, input_shape_to_nodes, output_shape_to_nodes, name_to_seq_num):
    trial_num = 0
    while True:
        if trial_num > 10:
            return None
        trial_num += 1
        print('searching...')
        input_shape = np.random.choice(list(input_shape_to_nodes.keys()))
        selected_node = np.random.choice(input_shape_to_nodes[input_shape])
        input_seq = name_to_seq_num[selected_node]
        output_nodes = [node for node in output_shape_to_nodes[input_shape] if name_to_seq_num[node] >= input_seq]
        if not output_nodes:
            continue
        output_node = np.random.choice(output_nodes)
        tmp_nodes = []
        assert name_to_seq_num[output_node] >= input_seq
        for i, node in enumerate(graph_def.node):
            if i < input_seq:
                continue
            n = _node_name(node.name)
            tmp_nodes.append(n)
            if i == name_to_seq_num[output_node]:
                break
        return tmp_nodes, input_shape


# import tensorflow.python.framework.graph_util_impl as utils

def remove_training_nodes(input_graph, protected_nodes=None):
    """Prunes out nodes that aren't needed for inference.
    There are nodes like Identity and CheckNumerics that are only useful
    during training, and can be removed in graphs that will be used for
    nothing but inference. Here we identify and remove them, returning an
    equivalent graph. To be specific, CheckNumerics nodes are always removed, and
    Identity nodes that aren't involved in control edges are spliced out so that
    their input and outputs are directly connected.
    Args:
      input_graph: Model to analyze and prune.
      protected_nodes: An optional list of names of nodes to be kept
        unconditionally. This is for example useful to preserve Identity output
        nodes.
    Returns:
      A list of nodes with the unnecessary ones removed.
    """
    if not protected_nodes:
        protected_nodes = []

    types_to_remove = {"CheckNumerics": True}

    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove and node.name not in protected_nodes:
            names_to_remove[node.name] = True

    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)
        nodes_after_removal.append(new_node)

    types_to_splice = {"Identity": True}
    names_to_splice = {}
    for node in nodes_after_removal:
        if node.op in types_to_splice and node.name not in protected_nodes:
            # We don't want to remove nodes that have control edge inputs, because
            # they might be involved in subtle dependency issues that removing them
            # will jeopardize.
            has_control_edge = False
            for input_name in node.input:
                if re.match(r"^\^", input_name):
                    has_control_edge = True
            if not has_control_edge:
                names_to_splice[node.name] = node.input[0]

    nodes_after_splicing = []
    for node in nodes_after_removal:
        if node.name in names_to_splice:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub(r"^\^", "", full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub(r"^\^", "", full_input_name)
            new_node.input.append(full_input_name)
        nodes_after_splicing.append(new_node)

    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph


def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def _extract_graph_summary(graph_def):
    """Extracts useful information from the graph and returns them."""
    name_to_input_name = {}  # Keyed by the dest node name.
    name_to_node = {}  # Keyed by node name.

    # Keeps track of node sequences. It is important to still output the
    # operations in the original order.
    name_to_seq_num = {}  # Keyed by node name.
    seq = 0
    for node in graph_def.node:
        n = _node_name(node.name)
        name_to_node[n] = node
        name_to_input_name[n] = [_node_name(x) for x in node.input]
        # Prevent colocated nodes from being lost.
        if "_class" in node.attr:
            for colocated_node_name in node.attr["_class"].list.s:
                colocated_node_decoded = colocated_node_name.decode("utf-8")
                if colocated_node_decoded.startswith("loc:@"):
                    name_to_input_name[n].append(colocated_node_decoded[5:])
        name_to_seq_num[n] = seq
        seq += 1
    return name_to_input_name, name_to_node, name_to_seq_num


def load_ph_file(pb_path):
    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    return sess.graph

def fill_layers(layer_with_nodes, name_to_node):
    first_node = name_to_node[layer_with_nodes[0]]

    if not first_node.input or len(first_node.input) == 1:
        return layer_with_nodes

    first_input = name_to_node[first_node.input[1].split(':')[0]]
    if first_input.op != 'Const':
        return layer_with_nodes

    for input in reversed(first_node.input):
        input_n = input.split(':')[0]
        if name_to_node[input_n].op == 'Const':
            layer_with_nodes.insert(0, input_n)
    return layer_with_nodes

if __name__ == '__main__':
    pb_path = '../models/densenet_tf_910.pb'
    graph = load_ph_file(pb_path)

    supported_ops_path = ''
    isolation_ops_path = 'isolation_ops'
    activation_ops_path = 'activation_ops'
    supported_ops_path = 'supported_ops'
    ignored_ops_path = ''
    isolation_ops = []
    activation_ops = []
    supported_ops = []
    with open(isolation_ops_path, 'r') as f:
        for line in f:
            isolation_ops.append(line.strip())
    with open(activation_ops_path, 'r') as f:
        for line in f:
            activation_ops.append(line.strip())
    with open(supported_ops_path, 'r') as f:
        for line in f:
            supported_ops.append(line.strip())

    graph_def = graph.as_graph_def()
    protected_ops = [graph.get_operations()[-1].node_def.name, graph.get_operations()[0].node_def.name]
    graph_def = remove_training_nodes(graph_def, protected_nodes=protected_ops)

    with tf.gfile.FastGFile(pb_path, mode='wb') as f:
        f.write(graph_def.SerializeToString())
    graph = load_ph_file(pb_path)

    selected_op = 'LA'

    name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(graph_def)

    input_shape_to_nodes = defaultdict(list)
    output_shape_to_nodes = defaultdict(list)

    for name, node in name_to_node.items():
        n = _node_name(name)
        op = graph.get_operation_by_name(n)
        if op.inputs and op.outputs:
            input_shape = op.inputs[0].shape[1:]
            output_shape = op.outputs[0].shape[1:]
            input_shape_to_nodes[str(input_shape)].append(n)
            output_shape_to_nodes[str(output_shape)].append(n)

    node_all = []
    if selected_op in ['LA']:
        selected_layer, layer_shape = get_selected_layer(graph_def, input_shape_to_nodes,
                                                         output_shape_to_nodes, name_to_seq_num)

        selected_layer = fill_layers(selected_layer, name_to_node)
        # if selected_layer is None:
        #     return False, None, None
        if selected_op == 'LA':
            suffix = '_LA'
            insert_points = input_shape_to_nodes[layer_shape]
            insert_layer = np.random.choice(insert_points)

        prev_node = insert_layer
        new_layer = []
        for index, name in enumerate(selected_layer):
            node = name_to_node[name]
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(node)
            new_node.name = new_node.name + suffix
            for i, node_input in enumerate(node.input):
                if node_input not in selected_layer and i == 0:
                    new_node.input[i] = prev_node
                    print('rebuld: link {} to {}'.format(prev_node, new_node.name))
                elif node_input in selected_layer:
                    new_node.input[i] = node_input + suffix
                new_layer.append(new_node)
            seq = name_to_seq_num[prev_node]
            insert_flag = False
            for i, node in enumerate(graph_def.node):
                if i == seq + 1:
                    print(i)
                    for new_node in new_layer:
                        node_all.append(new_node)
                    insert_flag = True
                if insert_flag and prev_node in node.input:
                    node.input[0] = new_node.name
                node_all.append(node)
        output_graph = graph_pb2.GraphDef()
        output_graph.node.extend(node_all)

        mutant_pb_path = '../models/mutate_pb.pb'
        with tf.gfile.FastGFile(mutant_pb_path, mode='wb') as f:
            f.write(output_graph.SerializeToString())


startup_message off
vbell off
autodetach on
defscrollback 1024
escape ^Zz

term xterm-256color

caption string "%?%F%{= Bk}%? %C%A %D %d-%m-%Y %{= kB} %t%= %?%F%{= Bk}%:%{= wk}%? %n "
hardstatus alwayslastline
hardstatus string '%{= kG}[ %{G}%H %{g}][%= %{= kw}%?%-Lw%?%{r}(%{W}%n*%f%t%?(%u)%?%{r})%{w}%?%+Lw%?%?%= %{g}][%{B} %d-%m-%Y %{W}%c:%s %{g}]'
