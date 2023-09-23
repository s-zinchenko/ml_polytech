import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix

from data import show_data, split_dataset
from model import KNN

if __name__ == '__main__':
    model = KNN()
    test_data, true_y, train_data = split_dataset(show_data(100), 0.7)

    model.train(train_data)
    result = model.predict(k=3, x_test=test_data)

    accuracy = accuracy_score(true_y, result,)
    confusion = confusion_matrix(true_y, result,)

    colors = {'red': 'r', 'blue': 'b', 'green': 'g'}
    # Вывод тренировочной выборки
    for point in train_data:
        plt.scatter(point[0], point[1], c=colors[point.y_color])

    # Вывод тестовой выборки
    for point, label in zip(test_data, result):
        plt.scatter(point[0], point[1], c=colors[label], edgecolors='k', s=100)
        # Вывод окружностей, включающих k ближайших соседей
        neighbors = model.get_kneighbors(point)
        for neighbor in neighbors:
            plt.gca().add_patch(plt.Circle(neighbor[2], 0.5, color='gray', fill=False))

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


# в тренировочной выборке цвет указываем
# в тестовой выборке цвета быть не должно
