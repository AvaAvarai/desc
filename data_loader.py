"""
Data loading module for various datasets.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_openml, load_breast_cancer


def load_iris_dataset():
    """Load the Fisher Iris dataset."""
    print("Loading Fisher Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    return X, y, feature_names, class_names


def load_mnist_dataset():
    """Load the MNIST dataset."""
    print("Loading MNIST dataset...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data
        y = mnist.target.astype(int)
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        class_names = np.array([str(i) for i in range(10)])
        return X, y, feature_names, class_names
    except Exception as e:
        raise Exception(f"Could not load MNIST dataset: {e}")


def load_wbc_30d_dataset():
    """Load the Wisconsin Breast Cancer 30-dimensional dataset."""
    print("Loading Wisconsin Breast Cancer dataset...")
    try:
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        feature_names = breast_cancer.feature_names
        class_names = breast_cancer.target_names
        return X, y, feature_names, class_names
    except Exception as e:
        raise Exception(f"Could not load Wisconsin Breast Cancer dataset: {e}")


def load_wbc_9d_dataset(csv_path='wbc9.csv'):
    """Load the Wisconsin Breast Cancer 9-dimensional dataset from CSV."""
    print(f"Loading Wisconsin Breast Cancer 9-dimensional dataset ({csv_path})...")
    try:
        wbc9_df = pd.read_csv(csv_path)
        X = wbc9_df.iloc[:, :-1].values
        y_wbc9_str = wbc9_df.iloc[:, -1].values
        unique_classes = np.unique(y_wbc9_str)
        class_names = unique_classes
        y = np.array([0 if label == 'Benign' else 1 for label in y_wbc9_str])
        feature_names = wbc9_df.columns[:-1].tolist()
        return X, y, feature_names, class_names
    except Exception as e:
        raise Exception(f"Could not load Wisconsin Breast Cancer 9D dataset: {e}")


def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        tuple: (X, y, feature_names, class_names) or None if dataset not found
    """
    dataset_loaders = {
        "Iris": load_iris_dataset,
        "MNIST": load_mnist_dataset,
        "Wisconsin Breast Cancer (30D)": load_wbc_30d_dataset,
        "Wisconsin Breast Cancer (9D)": load_wbc_9d_dataset,
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_loaders[dataset_name]()

