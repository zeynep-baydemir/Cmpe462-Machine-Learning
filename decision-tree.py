from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
best_accuracy = 0
best_depth = 0


for depth in max_depth:
    decision_tree = DecisionTreeClassifier(
        criterion="gini", max_depth=depth, random_state=10
    )

    decision_tree.fit(X_train, y_train)
    # y_pred = decision_tree.predict(X_test)

    accuracy = decision_tree.score(X_test, y_test)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
        best_tree = decision_tree
        print("Accuracy for decision tree:", accuracy, "depth: ", depth)
print("Best accuracy: ", best_accuracy, " with maximum depth:", best_depth)
plt.figure(figsize=(10, 5))
plot_tree(
    best_tree,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
)
plt.show()

feature_importances = best_tree.feature_importances_
feature_importance_df = pd.DataFrame(
    {"Feature": data.feature_names, "Importance": feature_importances}
)

feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

top_5_features = feature_importance_df.head(5)

selected_features = top_5_features["Feature"].values
X_train_selected = X_train[:, feature_importance_df["Feature"].isin(selected_features)]
X_test_selected = X_test[:, feature_importance_df["Feature"].isin(selected_features)]

num_important_features = [5, 10, 15, 20]

for features in num_important_features:
    top_features = feature_importance_df.head(features)
    selected_features = top_features["Feature"].values
    X_train_featured = X_train[
        :, feature_importance_df["Feature"].isin(selected_features)
    ]
    X_test_featured = X_test[
        :, feature_importance_df["Feature"].isin(selected_features)
    ]

    logistic_regression = LogisticRegression(max_iter=5000)
    logistic_regression.fit(X_train_featured, y_train)
    score = logistic_regression.score(X_test_featured, y_test)
    print("Max feature: ", features, "score: ", score)

random_forest = RandomForestClassifier(criterion="gini", random_state=10)
random_forest.fit(X_train, y_train)
train_scores = random_forest.score(X_train, y_train)
test_scores = random_forest.score(X_test, y_test)
print("Train score: ", train_scores)
print("Test score: ", test_scores)

tree_numbers = [i for i in range(1, 200)]
train_scores = []
test_scores = []
for number in tree_numbers:
    random_forest = RandomForestClassifier(
        criterion="gini", random_state=10, n_estimators=number
    )
    random_forest.fit(X_train, y_train)
    train_scores.append(random_forest.score(X_train, y_train))
    test_scores.append(random_forest.score(X_test, y_test))


plt.plot(tree_numbers, train_scores, label="Train Scores")
plt.plot(tree_numbers, test_scores, label="Test Scores")
plt.xlabel("Number of Trees")
plt.ylabel("Scores")
plt.title("Train and Test Scores vs. Number of Trees")
plt.legend()
plt.show()
