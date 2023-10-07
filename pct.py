import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        self.weights = np.zeros(n_features + 1)

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(self.weights[1:], x) + self.weights[0]
        return self.activation(z)

    def train(self, X, y):
        n_features = X.shape[1]
        self.initialize_weights(n_features)

        for _ in range(self.n_iterations):
            for i in range(X.shape[0]):
                x = np.insert(X[i], 0, 1)
                y_pred = self.predict(x)
                error = y[i] - y_pred
                self.weights[1:] += self.learning_rate * error * x[1:]
                self.weights[0] += self.learning_rate * error * x[0]

# Tạo dữ liệu mô phỏng (cổng logic OR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Khởi tạo và huấn luyện mô hình Perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Dự đoán
predictions = [perceptron.predict(np.insert(x, 0, 1)) for x in X]
print("Dự đoán:", predictions)
