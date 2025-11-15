from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def choose_model(model_type, all_hyperparams, random_state=42):
    '''
    model_type : string
    model_params : dict of parameters
    '''
    model_params = all_hyperparams.get(model_type, {}).copy()
    model_params['random_state'] = random_state

    if model_type == "LDA":
        model = LinearDiscriminantAnalysis(**model_params)
    elif model_type == "SVM":
        model = LinearSVC(**model_params)
    elif model_type == "GaussianNB":
        model = GaussianNB(**model_params)
    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier(**model_params)
    else:
        print(f"Warning: Model type '{model_type}' not recognized. Defaulting to LDA.")
        model = LinearDiscriminantAnalysis(**model_params)
    return model


'''
Avoid heavy models like :
- random forest
- k-NN
- MLP
'''