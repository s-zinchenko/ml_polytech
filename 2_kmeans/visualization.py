import matplotlib.pyplot as plt


def visualization_2d(cluster_content, x_index: int, y_index: int):
	k = len(cluster_content)

	plt.xlim(0, 1.25)
	plt.ylim(0, 1.25)



	# 1, 4

	colors = ['r', 'g', 'b']

	if x_index == 0:
		plt.xlabel('sepal length')
	elif x_index == 1:
		plt.xlabel('sepal width')
	elif x_index == 2:
		plt.xlabel('petal length')
	elif x_index == 3:
		plt.xlabel('petal width')

	if y_index == 0:
		plt.ylabel('sepal length')
	elif y_index == 1:
		plt.ylabel('sepal width')
	elif y_index == 2:
		plt.ylabel('petal length')
	elif y_index == 3:
		plt.ylabel('petal width')



	for i in range(k):
		for q in range(len(cluster_content[i])):
			x_coordinate = cluster_content[i][q][x_index]
			y_coordinate = cluster_content[i][q][y_index]
			plt.scatter(x_coordinate, y_coordinate, c=colors[i])

	plt.show()
