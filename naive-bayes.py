from ucimlrepo import fetch_ucirepo
import numpy as np


def calculate_mean(x):
    return np.mean(x, axis=0)


def calculate_var(x):
    return np.var(x, axis=0, ddof=1)


def calculate_pdf(x, mean, var):
    return 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


def separate_labels(data, labels):
    m_indices = np.where(labels == "M")[0]
    b_indices = np.where(labels == "B")[0]

    x_malignant = data[m_indices]
    x_benign = data[b_indices]

    return x_malignant, x_benign


def predict(x_m_mean, x_m_var, x_b_mean, x_b_var, x_m_prob, feature_data, label_data):
    count, true_count, false_count = 0, 0, 0
    predicts = []
    for x in feature_data:
        malign_prob = np.prod(calculate_pdf(x, x_m_mean, x_m_var)) * x_m_prob
        benign_prob = np.prod(calculate_pdf(x, x_b_mean, x_b_var)) * (1 - x_m_prob)
        if malign_prob > benign_prob:
            prediction = "M"
        else:
            prediction = "B"
        predicts.append(prediction)
        if prediction == label_data[count]:
            true_count += 1
        else:
            false_count += 1
        count += 1
    return predicts, true_count, false_count


def naive_bayes(x_test, y_test, x_train, y_train):
    x_malignant, x_benign = separate_labels(x_train, y_train)
    x_malignant_mean = calculate_mean(x_malignant)
    x_benign_mean = calculate_mean(x_benign)
    x_malignant_var = calculate_var(x_malignant)
    x_benign_var = calculate_var(x_benign)
    prob_x_malignant = x_malignant.shape[0] / x_train.shape[0]
    prob_x_benign = x_benign.shape[0] / x_train.shape[0]
    return predict(
        x_malignant_mean,
        x_malignant_var,
        x_benign_mean,
        x_benign_var,
        prob_x_malignant,
        x_test,
        y_test,
    )


if __name__ == "__main__":
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    np.random.seed(100)

    indices = np.random.permutation(y.shape[0])
    X_shuffled = X.values[indices]
    y_shuffled = y.values[indices]

    number_train = int(0.8 * X_shuffled.shape[0])
    train_data = X_shuffled[:number_train]
    test_data = X_shuffled[number_train:]
    train_labels = y_shuffled[:number_train]
    test_labels = y_shuffled[number_train:]

    pred_test, true_pred_test, false_pred_test = naive_bayes(
        test_data, test_labels, train_data, train_labels
    )
    accuracy_test = (true_pred_test / (true_pred_test + false_pred_test)) * 100
    print(accuracy_test)

    predictions_train, true_pred_train, false_pred_train = naive_bayes(
        train_data, train_labels, train_data, train_labels
    )
    accuracy_train = (true_pred_train / (true_pred_train + false_pred_train)) * 100
    print(accuracy_train)
