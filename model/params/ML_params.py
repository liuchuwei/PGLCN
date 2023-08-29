"""
modify from: https://github.com/marakeby/pnet_prostate_paper
"""

def require_ML_params(model_type):

    if model_type == "sgd":
        """
        L2 Logistic Regression
        """
        params = {'loss': 'log_loss', 'penalty': 'l2', 'alpha': 0.01}

    if model_type == "svc_rbf":
        """
        RBF Support Vector Machine
        """
        params = {'kernel': 'rbf', 'C': 100, 'gamma': 0.001, 'probability': True}

    if model_type == "svc_linear":
        """
        Linear Support Vector Machine
        """
        params = {'kernel': 'linear', 'C': 0.1, 'probability': True}

    if model_type == "random_forest":
        """
        Random Forest
        """
        params = {'max_depth': None, 'n_estimators': 50, 'bootstrap': False}

    if model_type == "adaboost":
        """
        Adaptive Boosting
        """
        params = {'learning_rate': 0.1, 'n_estimators': 50}


    if model_type == "decision_tree":
        """
        Decision Tree
        """
        params = {'min_samples_split': 10, 'max_depth': 10}


    return params