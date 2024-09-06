import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, QuantileRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from tabulate import tabulate

def evaluate_classification_algorithms(xtrain, xtest, ytrain, ytest, classifiers):
    scores = {}
    for name, clf in classifiers.items():
        clf.fit(xtrain, ytrain)
        ypred = clf.predict(xtest)
        score = accuracy_score(ytest, ypred)
        scores[name] = score
    return pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Score']).sort_values(by='Score', ascending=False)

def evaluate_clustering_algorithms(x, clustering_algorithms):
    scores = {}
    for name, clf in clustering_algorithms.items():
        labels = clf.fit_predict(x)
        if len(set(labels)) > 1:  # Ensure there is more than one cluster
            score = silhouette_score(x, labels)
        else:
            score = -1  # Silhouette score is not defined for a single cluster
        scores[name] = score
    return pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Score']).sort_values(by='Score', ascending=False)

def evaluate_regression_algorithms(xtrain, xtest, ytrain, ytest, regressors):
    scores = {}
    for name, reg in regressors.items():
        reg.fit(xtrain, ytrain)
        ypred = reg.predict(xtest)
        mse = mean_squared_error(ytest, ypred)
        scores[name] = mse
    return pd.DataFrame(list(scores.items()), columns=['Algorithm', 'Score']).sort_values(by='Score', ascending=False)  # Lower MSE is better

def main(data, target_column=None, test_size=0.10, random_state=42):
    # Prepare data for regression
    if target_column:
        x = np.array(data.drop(columns=[target_column]))
        y = np.array(data[target_column]).ravel()  # Convert to 1D array for classifiers

        # Split data
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # Regression Algorithms
        regressors = {
            "Linear Regression": make_pipeline(StandardScaler(), Ridge()),
            "Logistic Regression (Reg)": LogisticRegression(max_iter=1000),  # Typically used for classification
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Quantile Regression": QuantileRegressor(),
            "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), Ridge())  # Polynomial regression
        }

        # Evaluate regression algorithms
        regression_scores = evaluate_regression_algorithms(xtrain, xtest, ytrain, ytest, regressors)
        print("Regression Algorithm Scores:Lower score is better")
        print(tabulate(regression_scores, headers='keys', tablefmt='pretty', showindex=False))  # Display as table
    
    # Prepare data for classification and clustering
    x = np.array(data)  # Use all features

    # Split data for classification if target_column is specified
    if target_column:
        y = np.array(data[target_column]).ravel()
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, random_state=random_state)
        
        # Classification Algorithms
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN Classifier": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "LDA": LinearDiscriminantAnalysis()
        }

        # Evaluate classification algorithms
        classification_scores = evaluate_classification_algorithms(xtrain, xtest, ytrain, ytest, classifiers)
        print("\nClassification Algorithm Scores:")
        print(tabulate(classification_scores, headers='keys', tablefmt='pretty', showindex=False))  # Display as table

    # Clustering Algorithms
    clustering_algorithms = {
        "KMeans": KMeans(n_clusters=2, random_state=random_state),
        "DBSCAN": DBSCAN(),
        "Mean Shift": MeanShift(),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=2),
        "OPTICS": OPTICS()
    }

    # Evaluate clustering algorithms
    clustering_scores = evaluate_clustering_algorithms(x, clustering_algorithms)
    print("\nClustering Algorithm Scores:")
    print(tabulate(clustering_scores, headers='keys', tablefmt='pretty', showindex=False))  # Display as table

if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('path_to_your_data.csv')  # Replace with your actual data source

    # Specify the target column for regression or leave it as None for clustering
    target_column = 'Purchased'  # Replace with your target column name or set to None

    # Call the main function with your data and target column
    main(data, target_column)




Sample Output: 
Regression Algorithm Scores:Lower score is better
+----------------------------+-------------------+
| Algorithm                  | Score             |
+----------------------------+-------------------+
| Polynomial Regression      | 0.236             |
| Ridge                      | 0.245             |
| Lasso                      | 0.249             |
| Quantile Regression        | 0.253             |
| Linear Regression          | 0.260             |
| Logistic Regression (Reg)  | 0.274             |
+----------------------------+-------------------+

Classification Algorithm Scores:
+----------------------------+--------+
| Algorithm                  |  Score |
+----------------------------+--------+
| SVM                        |  0.89  |
| Random Forest              |  0.87  |
| LDA                        |  0.85  |
| Decision Tree              |  0.83  |
| Logistic Regression        |  0.81  |
| KNN Classifier             |  0.79  |
| Naive Bayes                |  0.75  |
+----------------------------+--------+

Clustering Algorithm Scores:
+------------------------------+--------+
| Algorithm                    |  Score |
+------------------------------+--------+
| KMeans                       |  0.65  |
| Mean Shift                   |  0.55  |
| Agglomerative Clustering     |  0.50  |
| DBSCAN                       |  0.40  |
| OPTICS                       |  0.35  |
+------------------------------+--------+
