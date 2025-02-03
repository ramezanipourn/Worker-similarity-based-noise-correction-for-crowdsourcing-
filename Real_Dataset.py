import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
from typing import Dict, List, Tuple
import logging
import arff
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
DATA_DIR = "data/real-world/"
T_NUM = 10

class WSNC:
    def __init__(self):
        pass

    def calculate_worker_similarity(self, worker_labels: Dict[str, Dict[int, int]]):
        worker_similarities = {}
        num_examples = len(next(iter(worker_labels.values())))

        for worker1, labels1 in worker_labels.items():
            similarities = []
            for worker2, labels2 in worker_labels.items():
                if worker1 == worker2:
                    similarity = 1.0
                else:
                    common = [(labels1[idx], labels2[idx]) for idx in labels1 if idx in labels2]
                    if len(common) >= num_examples * 0.1:
                        match_count = sum(1 for l1, l2 in common if l1 == l2)
                        similarity = match_count / len(common) if common else 0.0
                    else:
                        similarity = 0.0
                similarities.append((worker2, similarity))
            similarities.sort(key=lambda x: x[1], reverse=True)
            worker_similarities[worker1] = similarities

        return worker_similarities

    def find_knn(self, instance: pd.Series, dataset: pd.DataFrame, k: int):

        num_cols = dataset.select_dtypes(include=np.number).columns
        cat_cols = dataset.select_dtypes(exclude=np.number).columns

        max_vals = dataset[num_cols].max()
        min_vals = dataset[num_cols].min()
        distances = []

        for idx, row in dataset.iterrows():
            num_dist = np.sum([
                ((instance[col] - row[col]) / (max_vals[col] - min_vals[col])) ** 2
                if not (np.isnan(instance[col]) or np.isnan(row[col])) and (max_vals[col] != min_vals[col]) else 1
                for col in num_cols ])

            cat_dist = np.sum([1.0 for col in cat_cols if instance[col] != row[col]])
            total_dist = np.sqrt(num_dist + cat_dist)
            distances.append((idx, total_dist))
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:k]]

    def correct_noisy_labels(self, clean_dataset: pd.DataFrame, noisy_dataset: pd.DataFrame, predictions: pd.Series,integrated_labels):
        clean_dataset = clean_dataset.fillna({
            col: clean_dataset[col].mode()[0]
            if clean_dataset[col].dtype == 'object'
            else clean_dataset[col].mean()
            for col in clean_dataset.columns
        })

        noisy_dataset = noisy_dataset.fillna({
            col: noisy_dataset[col].mode()[0]
            if noisy_dataset[col].dtype == 'object'
            else noisy_dataset[col].mean()
            for col in noisy_dataset.columns
        })

        clean_X = clean_dataset.drop("class", axis=1)
        noisy_X = noisy_dataset.drop("class", axis=1)


        for col in clean_X.columns:
            if clean_X[col].dtype == 'object':
                le = LabelEncoder()
                clean_X[col] = le.fit_transform(clean_X[col].astype(str))
        for col in noisy_X.columns:
            if noisy_X[col].dtype == 'object':
                le = LabelEncoder()
                noisy_X[col] = le.fit_transform(noisy_X[col].astype(str))

        standardize = StandardScaler()
        if len(clean_X) > 0:
            clean_X = standardize.fit_transform(clean_X)
        if len(noisy_X) > 0:
            noisy_X = standardize.transform(noisy_X)

        if len(clean_X) > 0 and not pd.isnull(clean_X).all().all():
            dt_param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2', None]
            }

            dt_grid = GridSearchCV(
                DecisionTreeClassifier(),
                param_grid=dt_param_grid,
                cv=5,
                n_jobs=-1,
                scoring='accuracy'
            )
            dt_grid.fit(clean_X, clean_dataset['class'])
            clf1 = dt_grid.best_estimator_

            # برای GaussianNB
            nb_param_grid = {
                'var_smoothing': np.logspace(-12, -6, num=50)
            }

            nb_grid = GridSearchCV(
                GaussianNB(),
                param_grid=nb_param_grid,
                cv=5,
                n_jobs=-1,
                scoring='accuracy'
            )
            nb_grid.fit(clean_X, clean_dataset['class'])
            clf2 = nb_grid.best_estimator_
            clf1.fit(clean_X, clean_dataset['class'])
            clf2.fit(clean_X, clean_dataset['class'])

            pred1 = clf1.predict(noisy_X)
            pred2 = clf2.predict(noisy_X)
            acc_clf1 = dt_grid.best_score_
            acc_clf2 = nb_grid.best_score_


            for i, idx in enumerate(noisy_dataset.index):
                if pred1[i] == pred2[i] and pred1[i] != predictions[idx]:
                    integrated_labels[idx] = pred1[i]
                else:
                    integrated_labels[idx] = predictions[idx]

        return integrated_labels, acc_clf1, acc_clf2

class NoiseTester:
    def __init__(self):
        self.wsnc = WSNC()

    def majority_voting(self, worker_labels: Dict[str, Dict[int, int]], num_samples: int) -> List[int]:
        integrated_labels = []
        for idx in range(num_samples):
            label_counts = defaultdict(int)
            for worker, labels in worker_labels.items():
                if idx in labels:
                    label_counts[labels[idx]] += 1
            integrated_labels.append(max(label_counts, key=label_counts.get) if label_counts else -1)
        return integrated_labels



    def calculate_noise_ratio(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        incorrect_count = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != pred)
        return (incorrect_count / len(true_labels)) * 100 if len(true_labels) > 0 else 0.0

    def run_simulation(self, dataset: pd.DataFrame, worker_labels: Dict[str, Dict[int, int]], m_knn: int = 5):

        worker_similarities = self.wsnc.calculate_worker_similarity(worker_labels)
        predictions = pd.Series(index=dataset.index, dtype=int)
        integrated_labels = self.majority_voting(worker_labels, len(dataset))

        clean_examples, noisy_examples = [], []

        for idx in dataset.index:
            instance = dataset.iloc[idx].drop('class')
            knn_indices = self.wsnc.find_knn(instance, dataset.iloc[:,:-1], m_knn)

            label_confidence = defaultdict(float)
            label_totals = defaultdict(float)

            for neighbor_idx in knn_indices:
                for worker, labels in worker_labels.items():
                    if neighbor_idx in labels:
                        label = labels[neighbor_idx]
                        for sim_worker, sim_score in worker_similarities[worker][:m_knn]:
                            if sim_worker in worker_labels:
                                if neighbor_idx in worker_labels[sim_worker]:
                                    label_totals[label] += sim_score
                                    if worker_labels[sim_worker][neighbor_idx] == label:
                                        label_confidence[label] += sim_score

            for label in label_confidence:
                if label_totals[label] > 0:
                    label_confidence[label] /= label_totals[label]

            if len(label_confidence) > 0:
                predictions[idx] = max(label_confidence.items(), key=lambda x: x[1])[0]
            else:
                predictions[idx] = integrated_labels[idx]

            if predictions[idx] != integrated_labels[idx]:
                noisy_examples.append(idx)
            else:
                clean_examples.append(idx)

        clean_df = dataset.loc[clean_examples]
        noisy_df = dataset.loc[noisy_examples]


        corrected_predictions, acc_DecitionTree, acc_NB = self.wsnc.correct_noisy_labels(clean_df, noisy_df, predictions,integrated_labels)
        final_labels = corrected_predictions

        true_labels = dataset['class'].values
        noise_ratio = self.calculate_noise_ratio(true_labels, final_labels)
        return noise_ratio,  acc_DecitionTree, acc_NB
def load_dataset(name: str):
    response_path = os.path.join(DATA_DIR, name, f"{name}.response.txt")
    gold_path = os.path.join(DATA_DIR, name, f"{name}.gold.txt")
    arff_path = os.path.join(DATA_DIR, name, f"{name}.arff")


    with open(arff_path, 'r') as f:
        data = arff.load(f)
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

    worker_labels = defaultdict(dict)
    with open(response_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) < 3:
                logging.warning(f"Skipping malformed line: {line}")
                continue
            worker_id = parts[0]
            example_id = int(parts[1])
            label = int(parts[2])
            worker_labels[worker_id][example_id] = label

    true_labels = {}
    with open(gold_path, 'r') as f:
        for line in f:
            parts = line.strip().split('	')
            example_id = int(parts[0])
            label = int(parts[1])
            true_labels[example_id] = label

    df['class'] = df.index.map(true_labels)
    return df, worker_labels
def main():
    datasets = ["income94L10"] #"labelme"
    # kt_values = list(range(3, 10))
    kt_values = [5]

    tester = NoiseTester()
    for name in datasets:
        df, worker_labels = load_dataset(name)
        avg_noise_per_kt = []
        avg_acc_DT_per_kt = []
        avg_acc_NB_per_kt = []

        for KT in kt_values:
            total_noise = 0.0
            total_acc_DecisionTree = 0.0
            total_acc_NB = 0.0

            for _ in range(T_NUM):
                noise_ratio, acc_DecisionTree, acc_NB = tester.run_simulation(df, worker_labels, KT)
                total_noise += noise_ratio
                total_acc_DecisionTree += acc_DecisionTree
                total_acc_NB += acc_NB

            avg_noise = total_noise / T_NUM
            avg_acc_DecisionTree = total_acc_DecisionTree / T_NUM * 100
            avg_acc_NB = total_acc_NB / T_NUM * 100

            print(f'Result for KT : {KT}')
            print(f"|{name} average_noise : {avg_noise:.2f}")
            acc = (avg_acc_NB+avg_acc_DecisionTree)/2
            print(f"| average_acc : {acc:.2f}")


            avg_noise_per_kt.append(avg_noise)
            avg_acc_DT_per_kt.append(avg_acc_DecisionTree)
            avg_acc_NB_per_kt.append(avg_acc_NB)


        # plt.figure(figsize=(10, 6))
        # plt.plot(kt_values, avg_noise_per_kt, label="Avg Noise (%)", marker='o')
        # plt.title(f"Avg Noise vs KT for {name}")
        # plt.xlabel("KT (Number of Neighbors)")
        # plt.ylabel("Avg Noise (%)")
        # plt.grid(True)
        # plt.legend()
        # plt.savefig(f"{name}_avg_noise_vs_KT.png")
        # plt.close()

        print(f"Finished processing dataset {name}")
if __name__ == "__main__":
    main()


