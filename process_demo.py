import csv
import numpy as np
import tensorflow as tf

def get_edge(answer_path=None, truth_path=None):

    answer = csv.reader(open(answer_path, 'r'))
    object_set = []
    source_set = []
    object_index = []
    source_index = []
    claims = []
    for line in answer:
        object = line[0]
        source = line[1]

        if object not in object_set:
            object_set.append(object)
        if source not in source_set:
            source_set.append(source)

        object_index.append(object_set.index(object))
        source_index.append(source_set.index(source))
        claims.append(int(line[2]))

    object_index = np.array(object_index, dtype=np.int32)

    object_num = object_index.max() + 1

    source_index = np.array(source_index, dtype=np.int32) + object_num

    object_source_pair = np.vstack([object_index, source_index])

    claims = np.array(claims, dtype=np.int32)

    graph = {'object_source_pair':object_source_pair, 'claims':claims}

    # A = get_edge_index(object_source_pair)

    truths = get_truth(truth_path, object_set)

    return graph, object_index, source_index - object_num, truths

def get_truth(truth_path, object_set):
    obj_truth = csv.reader(open(truth_path, 'r'))
    truths = []
    gt_index = []
    for line in obj_truth:
        obj = line[0]
        if obj not in object_set:
            continue
        truth = int(line[1])
        truths.append(truth)
        gt_index.append(object_set.index(obj))

    return {'truths':np.array(truths), 'gt_index':np.array(gt_index)}

# def get_edge_index(object_source_pair):
#     row, col = object_source_pair
#     new_row = np.hstack([row, col])
#     new_col = np.hstack([col, row])
#     print(new_row)
#     print(new_col)
#     node_num = np.max(new_row) + 1
#     A = np.zeros(shape=(node_num, node_num))
#     A[new_row][new_col] == 1
#     # print(A)
#     return A

# dataset = 'duck'
# graph, object_index, source_index, truth_set = get_edge(answer_path='./zzfx/{}/answer.csv'.format(dataset),
#                                                         truth_path='./zzfx/{}/truth.csv'.format(dataset))
# print(graph['claims'])
# score = np.ones(shape=(len(graph['claims'])))
# print(tf.math.unsorted_segment_sum(data=score, segment_ids=graph['claims'], num_segments=9))
