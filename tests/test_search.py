from unittest.mock import patch

from autokeras.search import *
from autokeras import constant
import numpy as np


def simple_transform(_):
    generator = RandomConvClassifierGenerator(input_shape=(28, 28, 1), n_classes=3)
    return [Graph(generator.generate()), Graph(generator.generate())]


@patch('autokeras.search.transform', side_effect=simple_transform)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_hill_climbing_classifier_searcher(_, _1):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_MODEL_NUM = 10
    generator = HillClimbingSearcher(3, (28, 28, 1), verbose=False, path=constant.DEFAULT_SAVE_PATH)
    generator.search(x_train, y_train, x_test, y_test)
    assert len(generator.history) == len(generator.history_configs)


@patch('autokeras.search.ModelTrainer.train_model', side_effect=lambda: None)
def test_random_searcher(_):
    x_train = np.random.rand(2, 28, 28, 1)
    y_train = np.random.rand(2, 3)
    x_test = np.random.rand(1, 28, 28, 1)
    y_test = np.random.rand(1, 3)

    constant.MAX_MODEL_NUM = 3
    generator = RandomSearcher(3, (28, 28, 1), verbose=False, path=constant.DEFAULT_SAVE_PATH)
    generator.search(x_train, y_train, x_test, y_test)
    assert len(generator.history) == len(generator.history_configs)


# TODO: Test Bayesian Search

def test_search_tree():
    tree = SearchTree()
    tree.add_child(-1, 0)
    tree.add_child(0, 1)
    tree.add_child(0, 2)
    assert len(tree.adj_list) == 3
