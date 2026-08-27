"""
Decision Tree Classifier from scratch (NumPy + Matplotlib only)
================================================================

Goal of this example
---------------------
We build a small synthetic dataset that follows a clear, learnable
"checkerboard" pattern:

    label = 1   if (x1 > 5 and x2 > 5) or (x1 <= 5 and x2 <= 5)
    label = 0   otherwise

This is the classic XOR-style pattern: a single straight line cannot
separate the two classes, but a decision tree can, because it is
allowed to split the space more than once (first on x1, then on x2).

We then implement a decision tree classifier "by hand" (no sklearn),
train it on the data, and visualize the decision boundary it learned.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. Create the synthetic dataset
# ---------------------------------------------------------------
def make_pattern_data(n_samples=300, noise_ratio=0.05, seed=0):
    """Create 2D points in [0, 10] x [0, 10] with a checkerboard label pattern."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(0, 10, size=(n_samples, 2))

    # True underlying rule (the "pattern" the tree must discover)
    x1, x2 = X[:, 0], X[:, 1]
    y = ((x1 > 5) & (x2 > 5)) | ((x1 <= 5) & (x2 <= 5))
    y = y.astype(int)

    # Flip a small fraction of labels to make it a bit more realistic
    n_flip = int(n_samples * noise_ratio)
    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    return X, y


# ---------------------------------------------------------------
# 2. Impurity measure (Gini impurity)
# ---------------------------------------------------------------
def gini(y):
    """Gini impurity: 0 means the group is pure (all one class)."""
    if len(y) == 0:
        return 0.0
    p1 = np.mean(y)          # fraction of class 1
    p0 = 1.0 - p1            # fraction of class 0
    return 1.0 - p0 ** 2 - p1 ** 2


# ---------------------------------------------------------------
# 3. Find the best (feature, threshold) split
# ---------------------------------------------------------------
def best_split(X, y):
    """Search every feature and every candidate threshold for the split
    that gives the lowest weighted Gini impurity."""
    n_samples, n_features = X.shape
    best_feature, best_threshold, best_gini = None, None, np.inf

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            g_left = gini(y[left_mask])
            g_right = gini(y[right_mask])
            weighted_gini = (left_mask.sum() * g_left +
                              right_mask.sum() * g_right) / n_samples

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = t

    return best_feature, best_threshold, best_gini


# ---------------------------------------------------------------
# 4. Tree node + recursive tree building
# ---------------------------------------------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature      # which feature this node splits on
        self.threshold = threshold  # split threshold
        self.left = left            # Node for feature <= threshold
        self.right = right          # Node for feature > threshold
        self.label = label          # set only for leaf nodes


def build_tree(X, y, depth=0, max_depth=4, min_samples_split=5):
    # Stopping conditions -> make a leaf node
    if (depth >= max_depth
            or len(y) < min_samples_split
            or gini(y) == 0.0):
        majority_label = int(round(np.mean(y))) if len(y) > 0 else 0
        return Node(label=majority_label)

    feature, threshold, _ = best_split(X, y)

    if feature is None:  # no split improves things -> leaf
        majority_label = int(round(np.mean(y)))
        return Node(label=majority_label)

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    left_node = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples_split)
    right_node = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth, min_samples_split)

    return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)


# ---------------------------------------------------------------
# 5. Prediction
# ---------------------------------------------------------------
def predict_one(node, x):
    if node.label is not None:  # leaf node
        return node.label
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


# ---------------------------------------------------------------
# 6. Print the tree as text (helps students see what was learned)
# ---------------------------------------------------------------
def print_tree(node, depth=0):
    indent = "  " * depth
    if node.label is not None:
        print(f"{indent}Leaf -> class {node.label}")
    else:
        print(f"{indent}x[{node.feature}] <= {node.threshold:.2f} ?")
        print(f"{indent}True branch:")
        print_tree(node.left, depth + 1)
        print(f"{indent}False branch:")
        print_tree(node.right, depth + 1)


# ---------------------------------------------------------------
# 7. Visualization: data + learned decision boundary
# ---------------------------------------------------------------
def plot_decision_boundary(tree, X, y):
    x1_min, x1_max = 0, 10
    x2_min, x2_max = 0, 10

    grid_step = 0.1
    x1_grid, x2_grid = np.meshgrid(
        np.arange(x1_min, x1_max, grid_step),
        np.arange(x2_min, x2_max, grid_step),
    )
    grid_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    grid_predictions = predict(tree, grid_points).reshape(x1_grid.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(x1_grid, x2_grid, grid_predictions, alpha=0.3, levels=[-0.5, 0.5, 1.5],
                 colors=["#5b9bd5", "#ed7d31"])
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="#1f4e79", label="class 0", edgecolor="k")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="#c55a11", label="class 1", edgecolor="k")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision tree: learned decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 8. Run everything
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: make the pattern data
    X, y = make_pattern_data(n_samples=300, noise_ratio=0.05, seed=0)

    # Step 2: split into train / test sets (simple 80/20 split)
    rng = np.random.default_rng(1)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Step 3: train the decision tree
    tree = build_tree(X_train, y_train, max_depth=4)

    # Step 4: check accuracy on the held-out test set
    y_pred = predict(tree, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy * 100:.1f}%\n")

    # Step 5: print the learned tree structure
    print("Learned decision tree:")
    print_tree(tree)
    print()

    # Step 6: visualize the data and the decision boundary
    plot_decision_boundary(tree, X, y)
