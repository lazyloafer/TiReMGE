import numpy as np
import tensorflow as tf
from data.process_demo import get_edge
from utils import update_feature, get_adj, update_reliability, dis_loss, eval
import Model
import tensorflow.python.ops.numpy_ops.np_config as np_config

## time with CPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np_config.enable_numpy_behavior()

dataset='fej2013' # zc_all, zc_in, zc_us, sentiment, temp, uc, trec2011, cf, fej2013
graph, object_index, source_index, truth_set = get_edge(answer_path='./data/{}/answer.csv'.format(dataset),
                                                        truth_path='./data/{}/truth.csv'.format(dataset))
object_source_pair = graph['object_source_pair']
node_num = np.max(object_source_pair) + 1
object_num = np.max(object_source_pair[0]) + 1
source_num = node_num - object_num
class_num = int(np.max(truth_set['truths']) + 1)
claims = tf.one_hot(indices=graph['claims'], depth=class_num)

adj1, adj2 = get_adj(object_source_pair, node_num)
edge_index1 = adj1.astype(np.int32)
edge_index2 = adj2.astype(np.int32)

model = Model.TiReMGE(node_num=node_num, source_num=source_num, class_num=class_num)

learning_rate = 1e-2
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

reliability = np.ones(shape=(source_num)) / source_num
x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)

best_acc = 0

for step in range(150):
    with tf.GradientTape() as tape:
        embedding = model([x, edge_index1, edge_index2])

        reliability = update_reliability(embedding, [object_index, source_index, claims], source_num, reliability)
        loss1 = dis_loss(embedding, [object_index, source_index, claims], reliability, source_num)
        loss = loss1

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))
    x = update_feature([object_index, source_index, claims], reliability, object_num, source_num)

    acc = eval(embedding, truth_set, class_num)
    if acc > best_acc:
        best_acc = acc
    print("step = {}\tloss = {}\tbest_accuracy = {}\taccuracy = {}".format(step, loss, best_acc, acc))


