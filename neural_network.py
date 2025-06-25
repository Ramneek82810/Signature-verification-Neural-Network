import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=2500, hidden_size1=128, hidden_size2=64, output_size=1, learning_rate=0.1):
        np.random.seed(42)

        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))

        self.lr = learning_rate
        
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def train(self, X, y, epochs=1000, batch_size=128):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            total_loss=0

            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]

                # Forward pass
                z1 = np.dot(X_batch, self.W1) + self.b1
                a1 = self.leaky_relu(z1)

                z2 = np.dot(a1, self.W2) + self.b2
                a2 = self.leaky_relu(z2)

                z3 = np.dot(a2, self.W3) + self.b3
                a3 = self.sigmoid(z3)

                # Loss calculation
                epsilon = 1e-7
                loss = -np.mean(y_batch * np.log(a3 + epsilon) + (1 - y_batch) * np.log(1 - a3 + epsilon))
                total_loss+=loss

                # Backward pass
                d_z3 = a3 - y_batch
                d_W3 = np.dot(a2.T, d_z3) / batch_size
                d_b3 = np.sum(d_z3, axis=0, keepdims=True) / batch_size

                d_a2 = np.dot(d_z3, self.W3.T)
                d_z2 = d_a2 * self.leaky_relu_derivative(z2)
                d_W2 = np.dot(a1.T, d_z2) / batch_size
                d_b2 = np.sum(d_z2, axis=0, keepdims=True) / batch_size

                d_a1 = np.dot(d_z2, self.W2.T)
                d_z1 = d_a1 * self.leaky_relu_derivative(z1)
                d_W1 = np.dot(X_batch.T, d_z1) / batch_size
                d_b1 = np.sum(d_z1, axis=0, keepdims=True) / batch_size

                # Standard SGD Updates
                self.W3 -= self.lr * d_W3
                self.b3 -= self.lr * d_b3
                self.W2 -= self.lr * d_W2
                self.b2 -= self.lr * d_b2
                self.W1 -= self.lr * d_W1
                self.b1 -= self.lr * d_b1

            if epoch % 100 == 0:
                avg_loss = total_loss / (X.shape[0] / batch_size)
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.leaky_relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.leaky_relu(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.sigmoid(z3)

        return (a3 > 0.5).astype(int)
