import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC


data_dict = pickle.load(open('./data_new.pickle', 'rb'))
#print(data_dict['data'])
# Assuming data_dict['data'] contains lists of varying lengths
max_length = 42  # Choose a maximum length for padding

data_array = []
for item in data_dict['data']:
    # Pad shorter lists with zeros
    padded_item = np.pad(item[:max_length], (0, max_length - len(item[:max_length])), mode='constant')

    data_array.append(padded_item)

data = np.asarray(data_array)  # Now data will have a consistent shape

labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define SVM, DT, and GBM models
svm_model = SVC()
dt_model = RandomForestClassifier()
gbm_model = GradientBoostingClassifier()

# Create a VotingClassifier ensemble with all three models
ensemble_model = VotingClassifier(estimators=[('svm', svm_model), ('dt', dt_model)], voting='hard')

# Train the ensemble model
ensemble_model.fit(x_train, y_train)

# Predict using the ensemble model
y_predict = ensemble_model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_ensemble_newdata.p', 'wb')
pickle.dump({'model': ensemble_model}, f)
f.close()
