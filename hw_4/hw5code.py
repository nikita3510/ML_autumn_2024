from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    unique_values = np.unique(feature_vector)
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    ginis = np.zeros(len(thresholds))

    total_count = len(target_vector)
    total_positive = np.sum(target_vector)
    total_negative = total_count - total_positive

    for i, threshold in enumerate(thresholds):
        left_mask = feature_vector <= threshold
        right_mask = feature_vector > threshold

        if np.any(left_mask) and np.any(right_mask):
            left_positive = np.sum(target_vector[left_mask])
            left_negative = np.sum(left_mask) - left_positive
            right_positive = total_positive - left_positive
            right_negative = total_negative - left_negative

            H_left = 1 - (left_positive / (left_positive + left_negative)) ** 2 - (
                        left_negative / (left_positive + left_negative)) ** 2
            H_right = 1 - (right_positive / (right_positive + right_negative)) ** 2 - (
                        right_negative / (right_positive + right_negative)) ** 2

            gini_left = (left_mask.sum() / total_count) * H_left
            gini_right = (right_mask.sum() / total_count) * H_right

            ginis[i] = -gini_left - gini_right

    best_index = np.argmin(ginis)
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(sub_y) < self._min_samples_split or len(sub_y) <= self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: counts[key] / clicks[key] if key in clicks else 0 for key in counts}
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                categories_map = dict(zip([x[0] for x in sorted_categories], range(len(sorted_categories))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best if self._feature_types[feature_best] == "real" else None
        node["categories_split"] = threshold_best if self._feature_types[feature_best] == "categorical" else None
        node["left_child"], node["right_child"] = {}, {}

        if np.any(split):
            self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        if np.any(~split):
            self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_value = x[node["feature_split"]]

        if self._feature_types[node["feature_split"]] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            category_index = categories_map.get(feature_value)
            if category_index is not None and category_index < len(node["categories_split"]):
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = [self._predict_node(x, self._tree) for x in X]
        return np.array(predicted)