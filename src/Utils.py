import numpy as np
import math
import scipy.sparse as sp

adjStorage = {}
featureStorage = {}
labelStorage = []


def addEdge(graph, u, v):
    graph[u].append(v)


def getListOfGraphs():
    graphIndices = []
    ele = 0
    for line in open('../data/GraphLabelList'):
        graphIndices.append(ele)
        ele += 1

    return graphIndices


def loadCandidateSamples(sampled_index, features, support, y_train):
    yTrain = []
    for ele in sampled_index:
        yTrain.append(y_train[ele])
    return features, support, yTrain


def adjRaw():
    indOfGraph = 0
    vertices = 0
    mat = [[0] * vertices] * vertices

    for line in open('../data/GraphAdjList'):
        items = line.strip().split(' ')
        if items[0] == '-1' and items[1] == "-1":
            adjStorage[indOfGraph] = sp.csr_matrix(mat)
            indOfGraph += 1
        elif (items[0] == 'Size'):
            vertices = int(items[1])
            mat = [[0] * vertices] * vertices
        else:
            mat[int(items[0])][int(items[1])] = 1

    return adjStorage


def featureRaw():
    indOfGraph = 0
    indrow = 0
    rows = 0
    cols = 0
    mat = [[0] * cols] * rows

    for line in open('../data/GraphFeatureList'):
        items = line.strip().split(' ')
        if items[0] == '-1':
            featureStorage[indOfGraph] = sp.csr_matrix(mat)
            indOfGraph += 1
            indrow = 0
        elif (items[0] == 'Size'):
            rows = int(items[1])
            cols = int(items[2])
            mat = [[0] * cols] * rows
        else:
            for col in range(cols):
                mat[indrow][col] = float(items[col])

            indrow += 1

    return featureStorage


def labelsRaw():
    for line in open('../data/GraphLabelList'):
        items = line.strip().split(' ')
        labelStorage.append(int(items[0]))

    return labelStorage


def read_raw():
    adjRaw()
    featureRaw()
    labelsRaw()

    return adjStorage, featureStorage, labelStorage


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, numberOfChosenGraph, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels'][i]: labels[i] for i in range(len(labels))})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[0][1].shape})
    feed_dict.update({placeholders['numberOfChosenGraph']: numberOfChosenGraph})

    return feed_dict


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
