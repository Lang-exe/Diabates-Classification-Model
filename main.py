import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
from imblearn.over_sampling import RandomOverSampler

#Read Data"
diabetes_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Semester 4/Machine Learning/diabetes.csv')
diabetes_df.head()

#EDA (Exploratory Data Analysis)
diabetes_df.columns

diabetes_df.info()

diabetes_df.describe()

diabetes_df.describe().T

diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Showing the Count of NANs
print(diabetes_df_copy.isnull().sum())

diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].mean(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].mean(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].mean(), inplace = True)

print(diabetes_df_copy.isnull().sum())

# Remove Outliers
outliers = diabetes_df_copy.quantile(.97) # dealing with the outliers seen in the boxplots above

diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['Pregnancies']<outliers['Pregnancies'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['Glucose']<outliers['Glucose'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['BloodPressure']<outliers['BloodPressure'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['SkinThickness']<outliers['SkinThickness'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['Insulin']<outliers['Insulin'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['BMI']<outliers['BMI'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['DiabetesPedigreeFunction']<outliers['DiabetesPedigreeFunction'])]
diabetes_df_copy = diabetes_df_copy[(diabetes_df_copy['Age']<outliers['Age'])]

# Scalling the Data
diabetes_df_copy.head()

x = diabetes_df_copy.iloc[:,:-1]
x.head()

y = diabetes_df_copy.iloc[:, -1]
y.head()

col = x.columns
std = StandardScaler()

x = std.fit_transform(x)
x = pd.DataFrame(data = x, columns = col)

x.head()

# Class Balancing
print(y.value_counts())

over = RandomOverSampler()
x, y = over.fit_resample(x, y)

print(y.value_counts())

# Model Building
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_train = rfc.predict(X_train)
from sklearn import metrics

print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train)))

from sklearn import metrics

predictionsrfc = rfc.predict(X_test)
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictionsrfc)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictionsrfc))
print(classification_report(y_test,predictionsrfc))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

from sklearn import metrics

predictions = dtree.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

# Support Vector Machine
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)

from sklearn import metrics
svc_pred = svc_model.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))

# XgBoost Classifier
from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)

from sklearn import metrics

xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score =", format(metrics.accuracy_score(y_test, xgb_pred)))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test,xgb_pred))
