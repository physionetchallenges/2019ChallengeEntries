import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

# read the data
processed_df = pd.read_csv('D:/PycharmProjects/physionet-master/training_data/combined_imputed_scaled_data.csv')
column_names = processed_df.columns.values[1:]
X, y = processed_df[column_names[:-1]].values, processed_df[column_names[-1]].values

# shuffle the data
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)

# split the data
train_num = int(0.8*X.shape[0])
test_num = X.shape[0]-train_num
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_num, test_size=test_num)

# select feature by importance in ascend order
f_importance = ['chi_sort.npy', 'f_sort.npy', 'mut_inf_sort.npy', 'svc_sort.npy', 'clf_sort.npy']
# feature_num = [8, 9, 14, 20, 27, 30, 32]
print('X shape:', X.shape)
f_number = [9, 19, 29, 39]
for f_num in f_number:
    for f_imp in f_importance:
        t1 = time.time()
        print('feature importance:', f_imp)
        feature_imp = np.load(f_imp)
        X_train_sub, X_test_sub = X_train[:, feature_imp[f_num:]], X_test[:, feature_imp[f_num:]]
        # train logistic model
        clf = LinearSVC(C=1.0, class_weight='balanced', dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=10000,
         multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0).fit(X_train_sub, y_train)
        score = clf.score(X_test_sub, y_test)
        # print('Best C % .4f' % clf.C_)
        print("Coefficient:", clf.coef_)
        print("Test score with L1 penalty:",  score)
        print('elapsed time:', time.time()-t1)
    

# cross validation
'''
X_sub = X[:, feature_num]
# clf = svm.SVC(kernel='linear', C=1)
clf = LinearSVC(C=1.0, class_weight='balanced', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
scores = cross_val_score(clf, X_sub, y, cv=5)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
'''