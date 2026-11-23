from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import inspect

def choose_model(model_type, all_hyperparams, random_state=42):
    '''
    model_type : string
    model_params : dict of parameters
    '''
    model_class_map = {
        "LDA": LinearDiscriminantAnalysis,
        "SVM": LinearSVC,
        "GaussianNB": GaussianNB,
        "DecisionTree": DecisionTreeClassifier,
        "RandomForest": RandomForestClassifier,
    }
    
    model_class = model_class_map.get(model_type, LinearDiscriminantAnalysis)
    
    # Get valid parameters for this model
    valid_params = inspect.signature(model_class.__init__).parameters.keys()
    
    # Filter hyperparams to only valid ones
    model_params = {k: v for k, v in all_hyperparams.get(model_type, {}).items() 
                    if k in valid_params}
    
    # Add random_state only if the model accepts it
    if 'random_state' in valid_params:
        model_params['random_state'] = random_state
    
    model = model_class(**model_params)
    return model


'''
Avoid heavy models like :
- random forest
- k-NN
- MLP
'''