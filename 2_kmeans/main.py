from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

from clustering import KMeans
from visualization import visualization_2d


if __name__ == '__main__':
    iris = datasets.load_iris()
    scaler = MinMaxScaler()
    scaler.fit(iris.data)
    iris_data = scaler.transform(iris.data)

    model = KMeans()
    content = model.clustering(iris_data, 3)

    score = model.silhouette_score(iris_data, iris.target)
    print("Silhouette Score:", score)

    visualization_2d(content, 0, 1)
    visualization_2d(content, 0, 2)
    visualization_2d(content, 0, 3)
    visualization_2d(content, 1, 2)
    visualization_2d(content, 1, 3)
    visualization_2d(content, 2, 3)

