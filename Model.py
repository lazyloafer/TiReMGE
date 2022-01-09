import tf_geometric as tfg
import tensorflow as tf
import numpy as np

class TiReMGE(tf.keras.Model):
    def __init__(self, node_num, source_num, class_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gcn10 = tfg.layers.GCN(256, activation=tf.nn.relu)

        self.gcn20 = tfg.layers.GCN(64, activation=None)
        self.gcn21 = tfg.layers.GCN(256, activation=tf.nn.relu)

        self.fc1 = tf.keras.layers.Dense(class_num)

    def call(self, input, training=None):
        x, edge_index1, edge_index2 = input

        h1 = self.gcn10([x, edge_index1])

        h2 = self.gcn20([x, edge_index2])
        h2 = self.gcn21([h2, edge_index2])

        h = h1 + h2

        h = self.fc1(h)

        return h