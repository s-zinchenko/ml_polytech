import random
from typing import List, NamedTuple, Optional

import matplotlib.pyplot as plt


class Point(NamedTuple):
    x1: float
    x2: float
    y_color: Optional[str]


def generate_delimited_points(num_points):
    points: List[Point] = []

    for _ in range(num_points):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)

        if x1 < 3 and x2 < 3:
            points.append(Point(x1, x2, 'red'))
        elif x1 >= 3 and x1 < 6:
            points.append(Point(x1, x2, 'green'))
        elif x1 >= 6 and x2 >= 6:
            points.append(Point(x1, x2, 'blue'))

    return points


def split_dataset(data: List[Point], test_percent: float):
    train_data: List[Point] = []
    test_data: List[Point] = []
    test_data_true_y: List[str] = []
    for row in data:
        if random.random() < test_percent:
            test_data.append(Point(x1=row.x1, x2=row.x2, y_color=None))
            test_data_true_y.append(row.y_color)
        else:
            train_data.append(row)
    return test_data, test_data_true_y, train_data


def show_data(num_points: int):
    points = generate_delimited_points(num_points)

    for point in points:
        plt.scatter(point.x1, point.x2, color=point.y_color)

    plt.xlim(0, 10)
    plt.ylim(0, 10)

    plt.show()

    return points
