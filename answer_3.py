import csv
import numpy as np
from preprocessing import load_clean_normal_data, selected_features
from models import train_one_class_svm, train_elliptic_envelope, train_isolation_forest
import io

data = load_clean_normal_data()

models = \
       {'One_Class_SVM': train_one_class_svm(data)
        , 'Elliptical_Envelope': train_elliptic_envelope(data)
        , 'Isolation_Forrest': train_isolation_forest(data)
        }
for modelName, model in models.items():
    scores = model.decision_function(data[selected_features])
    sortedIndices = np.argsort(scores)
    sortedNames = data['Player'].iloc[sortedIndices]
    sortedScores = np.array(scores)[sortedIndices]
    sortedResults = zip(sortedNames,sortedScores)
    with open(modelName+'_Scores.csv','w', encoding="utf-8", newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(('Player','Result'))
        writer.writerows(sortedResults)
