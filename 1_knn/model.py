import math
from operator import itemgetter
from typing import List

import numpy as np

from data import Point


class KNN:
    def __init__(self, **kwargs):
        self.k = kwargs.get("k", None)
        self.kneighbors = {}

    def train(self, x_train: List[Point]):
        self._x_train = x_train
        self._y_train: List[str] = [point.y_color for point in x_train]

    def get_kneighbors(self, obj):
        if not self.k:
            raise Exception

        return self.kneighbors[obj]

    def eucl_dist(self, p_vector, q_vector) -> float:
        return math.sqrt(
            sum(
                [
                    (pi - qi) ** 2
                    for pi, qi in zip(p_vector, q_vector)
                ]
            )
        )

    def predict_for_single_obj(self, k: int, predictable_object: Point):
        self.k = k

        neighbors_list = []
        for i in range(len(self._x_train)):
            data_train_current_x = [self._x_train[i].x1, self._x_train[i].x2]
            data_train_current_y = self._y_train[i]
            dist = self.eucl_dist(predictable_object, data_train_current_x)
            temp_res = (data_train_current_y, dist, data_train_current_x)
            neighbors_list.append(temp_res)
        neighbors_list_sorted = sorted(neighbors_list, key=itemgetter(1))
        k_neighbors_list_sorted = neighbors_list_sorted[:k]

        self.kneighbors[predictable_object] = k_neighbors_list_sorted

        k_y_list = [y for y, _, _ in k_neighbors_list_sorted]
        k_y_list_grouped_temp = np.unique(k_y_list, return_counts=True)
        k_y_list_grouped = [[key, cnt] for key, cnt in zip(k_y_list_grouped_temp[0], k_y_list_grouped_temp[1])]

        k_y_list_grouped_sorted = sorted(k_y_list_grouped, key=itemgetter(1), reverse=True)

        return k_y_list_grouped_sorted[0][0]

    def predict(self, k: int, x_test):
        test_data_temp: List[Point] = []
        for i in range(len(x_test)):
            data_test_current_x = [x for x in x_test[i]]
            test_data_temp.append(Point(*data_test_current_x))
        return [
            self.predict_for_single_obj(k=k, predictable_object=i)
            for i in test_data_temp
        ]

