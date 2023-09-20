import numpy as np 
from sklearn import svm 

from .timer import timer

def get_svm_classifier(type_: str) :
    types_all = ('linear', 'rbf', 'poly');
    assert type_ in types_all;

    params = {
        'kernel': type_,
        'C' : 1.0,
        'random_state': 0,
        'decision_function_shape': 'ovr',
    };
    
    if type_ == 'rbf' :
        params = {**params, 'gamma': 0.7};
    else :
        params = {**params, 'degree': 3};        

    return svm.SVC(**params);


@timer
def fit_classifier(clf, X, y) :
    assert X.shape[0] == y.shape[0];
    clf.fit(X, y);

    pred = predict_classifier(clf, X);
    acc = (np.count_nonzero(pred == y) / y.size) * 100;
    print(f"Training accuracy = {acc:.2f} %");


def predict_classifier(clf, X) :
    return clf.predict(X);
    

def get_svs_w_classes(clf, class_id=None) :
    svs = clf.support_vectors_.astype(np.float32);
    pred_class_ids = predict_classifier(clf, svs);

    if class_id is None :
        return svs, pred_class_ids;
    
    mask = (pred_class_ids == class_id);
    if np.count_nonzero(mask) == 0 :
        return None;

    return svs[mask];
