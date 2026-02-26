import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print("Distribusi label:", np.unique(labels, return_counts=True))
print("Panjang fitur:", len(data[0]))

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.1, shuffle=True, stratify=labels
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly'.format(score*100))

with open('model.p', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan ke 'model.p'")