#%% 0. LIBRARY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)

from business import EDA

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder, MinMaxScaler, OneHotEncoder
    )
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report
    )

import statsmodels.api as sm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.ensemble import RandomForestClassifier


#%% I.1 DATA PREPROCESSING => None

###--- Import data & Explore ---###
data_path = 'ml_datasource.csv'

eda = EDA(data_path)
df = eda.load_processed_data()
df.head()
df.info()

eda.plot_cols()

###--- Examine multicollinearity ---###
# Check correlation heatmap
eda.plot_corr()

# Check variance inflation factor
eda.check_vif()

#%% I.2 DATA PREP

X_train_array, X_test_array, y_train_array, y_test_array = (
    eda.load_encoded_data(method='ordinal')
    )

# %% ----------------- TEST ---------------

target = ['purchased']
y = df[target]
X = df.drop(columns=target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=365, stratify=y
    )


encoder = OneHotEncoder(sparse_output=False)
X_train_country_array = encoder.fit_transform(
    X_train[['student_country']]
    )

X_train_nocountry_array = X_train.drop(columns='student_country').to_numpy()

X_train_array = np.concatenate((X_train_country_array, X_train_nocountry_array), axis=1)

#---> need to drop country column in X


#%% II. LOGISTIC REGRESSION

acc_baseline = (
    pd.Series(y_train_array.reshape(1, -1)[0])
    .value_counts(normalize=True)
    .max()
    )

X_train_array = sm.add_constant(X_train_array)
X_test_array = sm.add_constant(X_test_array)

log_rec = sm.Logit(y_train_array, X_train_array)
log_rec_results = log_rec.fit()

print(log_rec_results.summary())

y_train_pred = (
    (log_rec_results.predict(X_train_array) > 0.5)
    .astype(int)
    .reshape(-1, 1)
    )
acc_train = (y_train_pred == y_train_array).sum() / len(y_train_array)


y_test_pred = (
    (log_rec_results.predict(X_test_array) > 0.5)
    .astype(int)
    .reshape(-1, 1)
    )
acc_test = (y_test_pred == y_test_array).sum() / len(y_test_array)


print(f'Accuracy baseline: {acc_baseline*100:.2f} %')

print(f'Accuracy on training data: {acc_train*100:.2f} %')

print(f'Accuracy on test data: {acc_test*100:.2f} %')

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

#%% III. K-Nearest Neighbors Model

param_grid = {
    'n_neighbors': range(1, 51),
    'weights': ['uniform', 'distance']
    }

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search_knn.fit(X_train_array, y_train_array)

knn_clf = grid_search_knn.best_estimator_
grid_search_knn.best_score_

y_test_pred = knn_clf.predict(X_test_array).reshape(-1, 1)
acc_test = (y_test_pred == y_test_array).sum() / len(y_test_array)

print(f'Accuracy on test data: {acc_test*100:.2f} %')

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

report = classification_report(y_test, y_test_pred)
print(report)

#%% IV. Support Vector Machines Model

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_array)
X_test_scaled = scaler.transform(X_test_array)

svm = SVC()

param_grid = {
    'C': range(1, 11),
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
    }

grid_search_svm = GridSearchCV(SVC(), param_grid, cv=5)
grid_search_svm.fit(X_train_scaled, y_train_array)
                               
svm_clf = grid_search_svm.best_estimator_

y_test_pred = svm_clf.predict(X_test_scaled).reshape(-1, 1)
acc_test = (y_test_pred == y_test_array).sum() / len(y_test_array)

print(f'Accuracy on test data: {acc_test*100:.2f} %')

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

report = classification_report(y_test, y_test_pred)
print(report)

#%% V. Decision Trees Model

ccp_alphas = [0, 0.001, 0.002, 0.003, 0.004, 0.005]
acc_array = []

for ccp_alpha in ccp_alphas:
    pruned_tree = DecisionTreeClassifier(
        random_state=365,
        ccp_alpha=ccp_alpha
        )
    
    pruned_tree.fit(X_train_array, y_train_array)
    
    y_train_pred = pruned_tree.predict(X_train_array).reshape(-1, 1)
    acc_test = (y_train_pred == y_train_array).sum() / len(y_train_pred)
    acc_array.append(acc_test)
    print(ccp_alpha)
    print(acc_test)



dct_clf = DecisionTreeClassifier(
    random_state=365,
    ccp_alpha=0
    )

dct_clf.fit(X_train_array, y_train_array)

y_train_pred = dct_clf.predict(X_train_array).reshape(-1, 1)
acc_train = (y_train_pred == y_train_array).sum() / len(y_train_pred)


y_test_pred = dct_clf.predict(X_test_array).reshape(-1, 1)
acc_test = (y_test_pred == y_test_array).sum() / len(y_test_array)

print(f'Accuracy on test data: {acc_test*100:.2f} %')


fig, ax = plt.subplots(figsize=(25, 12))

plot_tree(
    decision_tree=dct_clf,
    feature_names=X_train.columns.to_list(),
    filled=True,
    rounded=True,
    proportion=True,
    max_depth=3,
    fontsize=12,
    ax=ax,
    );

plt.show()

report = classification_report(y_test, y_test_pred)
print(report)

#%% VI. Random Forest model

rf = RandomForestClassifier(
    n_estimators=100,
    ccp_alpha=0,
    random_state=365
    )

rf.fit(X_train_array, y_train_array)

y_test_pred = rf.predict(X_test_array).reshape(-1, 1)
acc_test = (y_test_pred == y_test_array).sum() / len(y_test_array)

print(f'Accuracy on test data: {acc_test*100:.2f} %')

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()


