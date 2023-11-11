import tensorflow as tf
import matplotlib.pyplot as plt


def model(x, k_0, k_1, b):
    return k_0 * x ** 2 + x * k_1 + b


if __name__ == '__main__':
    noise = tf.random.normal(shape=[100], stddev=0.2)

    x = tf.random.uniform(shape=[100], minval=0, maxval=10)

    k_0_true = 2
    k_1_true = 3
    b_true = 4

    y_real = model(x, k_0_true, k_1_true, b_true) + noise
    k_0 = tf.Variable(0.3)
    k_1 = tf.Variable(0.4)
    b = tf.Variable(0.5)
    y_predicted = model(x, k_0, k_1, b)
    loss = tf.reduce_mean(tf.square(y_real - y_predicted))
    EPOCHS = 10
    learning_rate = 0.0002

    for n in range(EPOCHS):
        with tf.GradientTape() as t:
            y_predicted = model(x, k_0, k_1, b)
            loss = tf.reduce_mean(tf.square(y_real - y_predicted))

        dk_0, dk_1, db = t.gradient(loss, [k_0, k_1, b])
        k_0.assign_sub(learning_rate * dk_0)
        k_1.assign_sub(learning_rate * dk_1)
        b.assign_sub(learning_rate * db)

    y_predicted = model(x, k_0, k_1, b)
    plt.scatter(x, y_real, s=2)
    plt.scatter(x, y_predicted, c='r', s=2)
    plt.show()
