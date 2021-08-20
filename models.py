from preprocessing import selected_features

def train_one_class_svm(data):
    from sklearn.svm import OneClassSVM
    return OneClassSVM(kernel='rbf').fit(data[selected_features])

def train_elliptic_envelope(data):
    from sklearn.covariance import EllipticEnvelope
    return EllipticEnvelope(contamination=0.05,random_state=42).fit(data[selected_features])

def train_isolation_forest(data):
    from sklearn.ensemble import IsolationForest
    return IsolationForest(contamination=0.05,random_state=42).fit(data[selected_features])
