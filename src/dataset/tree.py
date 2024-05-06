import pickle
import lzma
import os
import numpy as np
from math import exp, fsum
from nltk.tree import Tree
from copy import deepcopy


def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node


def load_hierarchy(dataset, data_dir):
    """
    Load the hierarchy corresponding to a given dataset.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        data_dir: The directory where the hierarchy files are stored.

    Returns:
        A nltk tree whose labels corresponds to wordnet wnids.
    """
    if dataset in ["tiered-imagenet-84", "tiered-imagenet-224"]:
        fname = os.path.join(data_dir, "tiered_imagenet_tree.pkl")
    elif dataset in ["ilsvrc12", "imagenet"]:
        fname = os.path.join(data_dir, "imagenet_tree.pkl")
    elif dataset in ["inaturalist19-84", "inaturalist19-224"]:
        fname = os.path.join(data_dir, "inaturalist19_tree.pkl")
    else:
        raise ValueError("Unknown dataset {}".format(dataset))

    with open(fname, "rb") as f:
        return pickle.load(f)


def load_distances(dataset, dist_type, data_dir):
    """
    Load the distances corresponding to a given hierarchy.

    Args:
        dataset: The dataset name for which the hierarchy should be loaded.
        dist_type: The distance type, one of ['jc', 'ilsvrc'].
        data_dir: The directory where the hierarchy files are stored.
        shuffle_distances: Create random hierarchy maintaining the same weights

    Returns:
        A dictionary of the form {(wnid1, wnid2): distance} for all precomputed
        distances for the hierarchy.
    """
    assert dist_type in ["ilsvrc", "jc"]

    if dataset in ["tiered-imagenet-224", "tiered-imagenet-84"]:
        dataset = "tiered-imagenet"
    elif dataset in ["ilsvrc12", "imagenet"]:
        dataset = "imagenet"
    elif dataset in ["inaturalist19-224", "inaturalist19-84"]:
        dataset = "inaturalist19"

    with lzma.open(os.path.join(data_dir, "{}_{}_distances.pkl.xz".format(dataset, dist_type).replace("-", "_")), "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":

    data_dist = load_distances("tiered-imagenet-224", "ilsvrc", "hierarchy_pkl")
    print(data_dist)