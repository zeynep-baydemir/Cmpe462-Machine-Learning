import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_logistic(X, w):
    probabilities = sigmoid(np.dot(X, w))
    predictions = np.where(probabilities > 0.5, 1, -1)
    return predictions


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy


class DataPreprocessor:
    def __init__(self, X, y, training_size=0.8, random_seed=1000):
        self.X = X
        self.y = y
        self.sample_size = X.shape[0]
        self.training_size = training_size
        self.random_seed = random_seed

    @staticmethod
    def standard_scaler(X):
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X_scaled = (X - means) / stds
        return X_scaled

    @staticmethod
    def l2_normalizer(X):
        l2_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / l2_norms
        return X_normalized

    def shuffle_data(self, X, y):
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.sample_size)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        return X_shuffled, y_shuffled

    def encode_labels(self, y):
        y_encoded = np.array([-1 if label == "Osmancik" else 1 for label in y])
        return y_encoded

    def split_data(self, X, y):
        number_train = int(self.training_size * self.sample_size)
        X_train = X[:number_train]
        X_test = X[number_train:]
        y_train = y[:number_train]
        y_test = y[number_train:]
        return X_train, X_test, y_train, y_test

    def preprocess_data(self):
        X_scaled = DataPreprocessor.standard_scaler(self.X)
        X_normalized = DataPreprocessor.l2_normalizer(X_scaled)
        X_shuffled, y_shuffled = self.shuffle_data(
            X_normalized.to_numpy(), self.y.to_numpy()
        )
        y_encoded = self.encode_labels(y_shuffled)
        X_train, X_test, y_train, y_test = self.split_data(X_shuffled, y_encoded)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        return X_train, X_test, y_train, y_test

    def five_fold_data(self, X, y, fold_number):
        X_scaled = DataPreprocessor.standard_scaler(self.X)
        X_normalized = DataPreprocessor.l2_normalizer(X_scaled).to_numpy()
        y_encoded = self.encode_labels(self.y.to_numpy())
        fold_size = self.sample_size // 5
        if fold_number == 0:
            X_train = X_normalized[fold_size:]
            X_test = X_normalized[:fold_size]
            y_train = y_encoded[fold_size:]
            y_test = y_encoded[:fold_size]
        elif fold_number == 1:
            X_train = np.concatenate(
                (X_normalized[:fold_size], X_normalized[fold_size * 2 :])
            )
            X_test = X_normalized[fold_size : fold_size * 2]
            y_train = np.concatenate(
                (y_encoded[:fold_size], y_encoded[fold_size * 2 :])
            )
            y_test = y_encoded[fold_size : fold_size * 2]
        elif fold_number == 2:
            X_train = np.concatenate(
                (X_normalized[: fold_size * 2], X_normalized[fold_size * 3 :])
            )
            X_test = X_normalized[fold_size * 2 : fold_size * 3]
            y_train = np.concatenate(
                (y_encoded[: fold_size * 2], y_encoded[fold_size * 3 :])
            )
            y_test = y_encoded[fold_size * 2 : fold_size * 3]
        elif fold_number == 3:
            X_train = np.concatenate(
                (X_normalized[: fold_size * 3], X_normalized[fold_size * 4 :])
            )
            X_test = X_normalized[fold_size * 3 : fold_size * 4]
            y_train = np.concatenate(
                (y_encoded[: fold_size * 3], y_encoded[fold_size * 4 :])
            )
            y_test = y_encoded[fold_size * 3 : fold_size * 4]
        else:
            X_train = X_normalized[: fold_size * 4]
            X_test = X_normalized[fold_size * 4 :]
            y_train = y_encoded[: fold_size * 4]
            y_test = y_encoded[fold_size * 4 :]
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        return X_train, X_test, y_train, y_test


class FullBatchGradientDescent:
    def __init__(self, X, y, w, reg_param=None, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.X = X
        self.y = y
        self.w = w
        self.sample_size = X.shape[0]
        self.feature_size = X.shape[1]
        if reg_param is None:
            self.reg_param = 0.1
        else:
            self.reg_param = reg_param

    def calculate_gradient(self, X, y, w):
        gradient = 0
        loss_function = 0
        for i in range(self.sample_size):
            x_i = X[i].reshape(-1, 1)
            y_i = y[i]
            loss_function += np.log(1 + np.exp(-y_i * np.dot(w.T, x_i)))
            sigmoid_result = sigmoid(-y_i * np.dot(w.T, x_i))
            gradient = gradient - y_i * x_i * sigmoid_result
        gradient = -gradient / self.sample_size
        loss_function = (1 / self.sample_size) * loss_function
        return gradient, loss_function

    def calculate_gradient_with_regularization(self, X, y, w, reg_param):
        gradient = 0
        for i in range(self.sample_size):
            x_i = X[i].reshape(-1, 1)
            y_i = y[i]
            sigmoid_result = sigmoid(-y_i * np.dot(w.T, x_i))
            gradient = gradient - y_i * x_i * sigmoid_result
        gradient = -gradient / self.sample_size
        gradient += reg_param * w
        return gradient

    def calculate_weights(self):
        w = self.w
        for i in range(self.iterations):
            gradient, loss_function = self.calculate_gradient(self.X, self.y, w)
            w = w + self.learning_rate * gradient
            loss_values[i] = loss_function
        return w

    def calculate_weights_with_regularization(self):
        w = self.w
        for _ in range(self.iterations):
            gradient = self.calculate_gradient_with_regularization(
                self.X, self.y, w, self.reg_param
            )
            w = w + self.learning_rate * gradient
        return w


class StochasticGradientDescent:
    def __init__(self, X, y, w, reg_param=None, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.X = X
        self.y = y
        self.w = w
        self.sample_size = X.shape[0]
        if reg_param is None:
            self.reg_param = 0
        else:
            self.reg_param = reg_param

    def calculate_gradient(self, x_i, y_i, w):
        sigmoid_result = sigmoid(-y_i * np.dot(w.T, x_i))
        gradient = -y_i * x_i * sigmoid_result
        loss_function = np.log(1 + np.exp(-y_i * np.dot(w.T, x_i)))
        return gradient, loss_function

    def calculate_gradient_with_regularization(self, x_i, y_i, w, reg_param):
        sigmoid_result = sigmoid(-y_i * np.dot(w.T, x_i))
        gradient = -y_i * x_i * sigmoid_result
        gradient += reg_param * w
        return gradient

    def calculate_weights(self):
        w = self.w
        sample_size = self.X.shape[0]
        for j in range(self.iterations):
            loss_function = 0
            for t in range(self.sample_size):
                i = np.random.randint(0, sample_size)
                x_i = self.X[i].reshape(-1, 1)
                y_i = self.y[i]
                gradient, loss_value = self.calculate_gradient(x_i, y_i, w)
                loss_function += loss_value
                w = w - self.learning_rate / (t + 1) * gradient
            if self.learning_rate == 1:
                loss_values_large_lr[j] = (1 / self.sample_size) * loss_function
            elif self.learning_rate == 0.001:
                loss_values_large_sm[j] = (1 / self.sample_size) * loss_function
            else:
                loss_values2[j] = (1 / self.sample_size) * loss_function
        return w

    def calculate_weights_with_regularization(self):
        w = self.w
        sample_size = self.X.shape[0]
        for _ in range(self.iterations):
            i = np.random.randint(0, sample_size)
            x_i = self.X[i].reshape(-1, 1)
            y_i = self.y[i]
            gradient = self.calculate_gradient_with_regularization(
                x_i, y_i, w, self.reg_param
            )
            w = w - self.learning_rate * gradient
        return w


if __name__ == "__main__":
    rice_cammeo_and_osmancik = fetch_ucirepo(id=545)

    X = rice_cammeo_and_osmancik.data.features
    y = rice_cammeo_and_osmancik.data.targets

    preprocessor = DataPreprocessor(X, y)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    initial_weights = np.random.randn(X_train.shape[1], 1).reshape(-1, 1)

    loss_values = {}
    full_batch_gd = FullBatchGradientDescent(X_train, y_train, initial_weights)
    weights = full_batch_gd.calculate_weights()
    y_pred = predict_logistic(X_test, weights)
    accuracy = calculate_accuracy(y_test, y_pred)

    loss_values2 = {}
    stochastic_gd = StochasticGradientDescent(
        X_train, y_train, initial_weights, learning_rate=0.01
    )
    stochastic_weights = stochastic_gd.calculate_weights()
    y_pred_stochastic = predict_logistic(X_test, stochastic_weights)
    accuracy_stochastic = calculate_accuracy(y_test, y_pred_stochastic)

    x = list(loss_values.keys())
    y = [value[0][0] for value in loss_values.values()]
    y2 = [value[0][0] for value in loss_values2.values()]
    plt.plot(x, y, color="blue", label="Loss Function with Iteration")
    plt.plot(x, y2, color="red", label="Loss Function with Epoch")
    plt.title("Loss Function")
    plt.show()

    loss_values_large_lr = {}
    stochastic_large = StochasticGradientDescent(
        X_train, y_train, initial_weights, learning_rate=1
    )
    stochastic_weights_large = stochastic_large.calculate_weights()
    y_pred_stochastic_large = predict_logistic(X_test, stochastic_weights_large)
    accuracy_stochastic_large = calculate_accuracy(y_test, y_pred_stochastic_large)
    loss_values_large_sm = {}
    stochastic_small = StochasticGradientDescent(
        X_train, y_train, initial_weights, learning_rate=0.001
    )
    stochastic_weights_small = stochastic_small.calculate_weights()
    y_pred_stochastic_small = predict_logistic(X_test, stochastic_weights_small)
    accuracy_stochastic_small = calculate_accuracy(y_test, y_pred_stochastic_small)

    x2 = list(loss_values_large_lr.keys())
    y3 = [value[0][0] for value in loss_values_large_lr.values()]
    y4 = [value[0][0] for value in loss_values_large_sm.values()]
    plt.plot(x2, y3, color="blue", label="Large LR")
    plt.plot(x2, y4, color="red", label="Small LR")
    plt.title("Loss Function for LR")
    plt.show()

    print(f"Full Batch Gradient Descent Test Accuracy: {accuracy}")
    print(f"Stochastic Gradient Descent Test Accuracy: {accuracy_stochastic}")

    full_batch_gd_train = FullBatchGradientDescent(X_train, y_train, initial_weights)
    weights_train = full_batch_gd_train.calculate_weights()
    y_pred_train = predict_logistic(X_train, weights_train)
    accuracy_train = calculate_accuracy(y_train, y_pred_train)

    stochastic_gd_train = StochasticGradientDescent(X_train, y_train, initial_weights)
    stochastic_weights_train = stochastic_gd_train.calculate_weights()
    y_pred_stochastic_train = predict_logistic(X_train, stochastic_weights_train)
    accuracy_stochastic_train = calculate_accuracy(y_train, y_pred_stochastic_train)

    print(f"Full Batch Gradient Descent Train Accuracy: {accuracy_train}")
    print(f"Stochastic Gradient Descent Train Accuracy: {accuracy_stochastic_train}")

    reg_params = [0.0001, 0.0005, 0.001, 0.01]

    best_reg_param_fb = 0
    best_accuracy_fb = 0
    best_reg_param_st = 0
    best_accuracy_st = 0
    average_accuracy_fb = []
    average_accuracy_st = []
    average_accuracy_fb_train = []
    average_accuracy_st_train = []
    for param in reg_params:
        acc_fb = 0
        acc_st = 0
        acc_fb_train = 0
        acc_st_train = 0
        for i in range(5):
            X_train_folded, X_test_folded, y_train_folded, y_test_folded = (
                preprocessor.five_fold_data(X, y, i)
            )
            initial_weights = np.random.randn(X_train_folded.shape[1], 1).reshape(-1, 1)
            full_batch = FullBatchGradientDescent(
                X_train_folded, y_train_folded, initial_weights, param
            )
            fb_weights = full_batch.calculate_weights_with_regularization()
            fb_y = predict_logistic(X_test_folded, fb_weights)
            fb_accuracy = calculate_accuracy(y_test_folded, fb_y)
            acc_fb += fb_accuracy

            fb_y_train = predict_logistic(X_train_folded, fb_weights)
            fb_accuracy_train = calculate_accuracy(y_train_folded, fb_y_train)
            acc_fb_train += fb_accuracy_train

            stochastic = StochasticGradientDescent(
                X_train_folded, y_train_folded, initial_weights, param
            )
            stochastic_w = stochastic.calculate_weights_with_regularization()
            y_pred_st = predict_logistic(X_test_folded, stochastic_w)
            accuracy_st = calculate_accuracy(y_test_folded, y_pred_st)
            acc_st += accuracy_st

            y_pred_st_train = predict_logistic(X_train_folded, stochastic_w)
            accuracy_st_train = calculate_accuracy(y_train_folded, y_pred_st_train)
            acc_st_train += accuracy_st_train

        average_acc_st = acc_st / 5
        average_acc_fb = acc_fb / 5
        average_acc_st_train = acc_st_train / 5
        average_acc_fb_train = acc_fb_train / 5

        average_accuracy_fb.append(average_acc_fb)
        average_accuracy_st.append(average_acc_st)
        average_accuracy_fb_train.append(average_acc_fb_train)
        average_accuracy_st_train.append(average_acc_st_train)

    max_fb_acc = max(average_accuracy_fb)
    max_fb_index = average_accuracy_fb.index(max_fb_acc)
    max_st_acc = max(average_accuracy_st)
    max_st_index = average_accuracy_st.index(max_st_acc)
    max_fb_train_index = average_accuracy_fb_train.index(max(average_accuracy_fb_train))
    max_st_train_index = average_accuracy_st_train.index(max(average_accuracy_st_train))
    print(
        f"Full Batch Gradient Descent With Regularization Test Accuracy: {max_fb_acc} Parameter: {reg_params[max_fb_index]}"
    )
    print(
        f"Stochastic Gradient Descent With Regularization Test Accuracy: {max_st_acc} Parameter: {reg_params[max_st_index]}"
    )
    print(
        f"Full Batch Gradient Descent With Regularization Training Accuracy: {average_accuracy_fb_train[max_fb_index]} Parameter: {reg_params[max_fb_index]}"
    )
    print(
        f"Stochastic Gradient Descent With Regularization Training Accuracy: {average_accuracy_st_train[max_st_index]} Parameter: {reg_params[max_st_index]}"
    )
