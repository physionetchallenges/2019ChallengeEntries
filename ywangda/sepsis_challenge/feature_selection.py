from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, ShuffleSplit


processed_df = pd.read_csv('D:/PycharmProjects/physionet-master/training_data/combined_imputed_scaled_data.csv')
column_names = processed_df.columns.values[1:]
X, y = processed_df[column_names[:-1]].values, processed_df[column_names[-1]].values

# univariate feature selection
chi_1, pval_1 = chi2(X, y)
F_2, pval_2 = f_classif(X, y)
mut_inf = mutual_info_classif(X, y)
feature_imp1 = np.argsort(chi_1, axis=-1, kind='quicksort', order=None)
feature_imp2 = np.argsort(F_2, axis=-1, kind='quicksort', order=None)
feature_imp3 = np.argsort(mut_inf, axis=-1, kind='quicksort', order=None)

# correlation analysis
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
feature_imp4 = np.argsort(lsvc.coef_, axis=-1, kind='quicksort', order=None)

# pca unsupervised
from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
explained_variance_ = pca.explained_variance_

# tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
feature_imp5 = np.argsort(clf.feature_importances_, axis=-1, kind='quicksort', order=None)
print(clf.feature_importances_)

np.save('chi_sort', feature_imp1)
np.save('f_sort', feature_imp2)
np.save('mut_inf_sort', feature_imp3)
np.save('svc_sort', feature_imp4)
np.save('clf_sort', feature_imp5)