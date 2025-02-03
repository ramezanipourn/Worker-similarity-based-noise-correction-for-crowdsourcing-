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

                    common_indices = [idx for idx in labels1
                                      if idx in labels2
                                      and labels1[idx] != -1
                                      and labels2[idx] != -1]

                    if len(common_indices) >= num_examples * 0.1:
                        match_count = sum(1 for idx in common_indices
                                          if labels1[idx] == labels2[idx])
                        similarity = match_count / len(common_indices)
                    else:
                        similarity = 0.0
                similarities.append((worker2, similarity))
            similarities.sort(key=lambda x: x[1], reverse=True)
            worker_similarities[worker1] = similarities
        return worker_similarities



    def find_knn(self, instance: pd.Series, dataset: pd.DataFrame, k: int):
        num_cols = dataset.select_dtypes(include=np.number).columns
        cat_cols = dataset.select_dtypes(exclude=np.number).columns


        instance_filled = instance.copy()
        for col in num_cols:
            if pd.isna(instance[col]):
                instance_filled[col] = dataset[col].mean()
        for col in cat_cols:
            if pd.isna(instance[col]):
                instance_filled[col] = dataset[col].mode()[0]


        num_dist = np.sum([
            ((instance_filled[col] - dataset[col].fillna(dataset[col].mean())) /
             (dataset[col].max() - dataset[col].min())) ** 2 for col in num_cols], axis=0)



        cat_dist = np.sum([
            (dataset[col] != instance_filled[col]).astype(int)
            for col in cat_cols
        ], axis=0)

        total_dist = np.sqrt(num_dist + cat_dist)
        nearest_indices = total_dist.argsort()[:k]
        return dataset.iloc[nearest_indices].index.tolist()

    def correct_noisy_labels(self, clean_dataset: pd.DataFrame, noisy_dataset: pd.DataFrame, predictions: pd.Series,
                             integrated_labels):

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


        clean_y = clean_dataset['class']
        noisy_y = noisy_dataset['class']

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

        # مقیاس‌دهی ویژگی‌ها
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

            clf1.fit(clean_X, clean_y)
            clf2.fit(clean_X, clean_y)

            pred1 = clf1.predict(noisy_X)
            pred2 = clf2.predict(noisy_X)

            for i, idx in enumerate(noisy_dataset.index):
                if pred1[i] == pred2[i] and pred1[i] != predictions[idx]:
                    integrated_labels[idx] = pred1[i]
                else:
                    integrated_labels[idx] = predictions[idx]

        return integrated_labels

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

    def run(self, dataset: pd.DataFrame, worker_labels: Dict[str, Dict[int, int]], m_knn: int = 5):

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

        corrected_predictions = self.wsnc.correct_noisy_labels(clean_df, noisy_df, predictions,integrated_labels)
        final_labels = corrected_predictions

        true_labels = dataset['class'].values
        noise_ratio = self.calculate_noise_ratio(true_labels, final_labels)
        return noise_ratio


def simulate_workers(data: pd.DataFrame, num_workers: int = 11, pr_range: tuple = (0.55, 0.75)):
    true_labels = data["class"].values
    num_instances = len(true_labels)
    worker_labels = np.zeros((num_instances, num_workers), dtype=object)
    unique_labels = np.unique(true_labels)


    # worker_qualities = np.clip(np.random.normal(
    #     loc=(pr_range[0] + pr_range[1]) / 2,
    #     scale=(pr_range[1] - pr_range[0]) / 6,
    #     size=num_workers
    # ), pr_range[0], pr_range[1])

    worker_qualities = np.clip(
        np.random.normal(
            loc=(pr_range[0] + pr_range[1]) / 2,
            scale=(pr_range[1] - pr_range[0]) / 6,
            size=num_workers
        ),
        pr_range[0], pr_range[1])


    for worker_id in range(num_workers):
        for idx, label in enumerate(true_labels):
            if np.random.rand() < worker_qualities[worker_id]:
                worker_labels[idx, worker_id] = label
            else:

                class_probs = np.bincount(true_labels) / len(true_labels)
                worker_labels[idx, worker_id] = np.random.choice(unique_labels, p=class_probs)

    return pd.DataFrame(worker_labels, columns=[f"worker_{i + 1}" for i in range(num_workers)])

def load_dataset(name):

    arff_path = f"data\\synthetic\\{name}\\{name}.arff"
    with open(arff_path) as f:
        arff_data = arff.load(f)

    data = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
    data["class"] = data["class"].astype(int)
    return data

def run_simulation(dataset: pd.DataFrame, num_workers: int = 11, m_knn: int = 5, n_runs: int = 10):
    total_noise = 0.0
    for _ in range(n_runs):
        worker_labels = simulate_workers(dataset, num_workers=num_workers)
        worker_labels= worker_labels.to_dict('dict')
        simu = NoiseTester()
        noise_ratio = simu.run(dataset, worker_labels, m_knn)
        total_noise += noise_ratio

    return (total_noise/n_runs)


if __name__ == "__main__":
    names = [
        "anneal", "audiology", "autos", "balance-scale", "biodeg", "breast-cancer",
        "breast-w", "car", "credit-a", "credit-g", "diabetes", "heart-c",
        "heart-h", "heart-statlog", "hepatitis", "horse-colic", "hypothyroid",
        "ionosphere", "iris", "kr-vs-kp", "labor", "lymph", "mushroom", "segment",
        "sick", "sonar", "spambase", "tic-tac-toe", "vehicle", "vote", "vowel",
        "waveform", "zoo", "letter"
    ]
    name = 'sonar'
    dataset = load_dataset(name)
    noise_ratio = run_simulation(dataset)
    print(f'{name} dataset ratio_nois : {noise_ratio:.2f}')


