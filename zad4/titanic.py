"""
The program uses supervised learning to predict, based on data, whether the Titanic catastrophe passenger will survive.
Classification algorithms used:
    Decision Tree
    SVM
    Random Forest

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    libraries to install: sklearn, pandas, matplotlib
    to install  use command: "pip install <name of library >"

Preparing data:
    Most of the columns have been removed because they can interfere with the effectiveness of the model.
    Rows that do not contain the passenger's age have been deleted.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def draw_decision_tree(tree, headers, class_name, title):
    """
    The function that draw decision tree
    Parameters:
            tree (object): Algorithm used to create tree.
            headers (list): Headers of data that are used to training decision tree
            class_name (list): Names of classes that split data.
            title (str): Title of an image.
    """
    plt.figure(figsize=(15, 10))
    plot_tree(tree, filled=True, feature_names=headers, class_names=class_name, rounded=True)
    plt.title(title)
    plt.show()


'''data preparation'''
df = pd.read_csv('data/titanic.csv')
df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId', 'Fare', 'SibSp', 'Parch', 'Embarked'], inplace=True)
df.drop(df.loc[df.Age.isnull()].index, inplace=True)
df['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
df['Age'] = df['Age'].astype(int)

headers = df.columns

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

'''Split data to training and test set'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

'''Decision tree classification'''
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

y_pred_decision_tree = decision_tree.predict(X_test)

'''Standardization of features'''
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

svc = SVC(kernel='rbf', C=1.2, gamma=1)
svc.fit(X_train_std, y_train)
y_pred_svc = svc.predict(X_test_std)

'''Random Forest Classification'''
random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_test)

'''Image of decision tree '''
draw_decision_tree(decision_tree, headers[:-1], ['0', '1', '2'], "Decision Tree")

'''Image example of decision tree created by random forest tree algorithm (last tree created by random forest)'''
draw_decision_tree(random_forest.estimators_[-1], headers[:-1], ['0', '1', '2'], 'Random Forest Tree')

decision_tree_accuracy = round(accuracy_score(y_test, y_pred_decision_tree), 4)
random_forest_accuracy = round(accuracy_score(y_test, y_pred_random_forest), 4)
svm_accuracy = round(accuracy_score(y_test, y_pred_svc), 4)

print('Decision tree accuracy:', decision_tree_accuracy)
print('Random forest accuracy:', random_forest_accuracy)
print('SVM accuracy:', svm_accuracy)

classifier = ['Decision Tree', 'Random Forest', 'SVM']
classifier_accuracy = [decision_tree_accuracy, random_forest_accuracy, svm_accuracy]

plt.bar(classifier, classifier_accuracy)
plt.ylabel('accuracy')
for i, value in enumerate(classifier_accuracy):
    plt.text(i, 0.5, str(value), ha='center', va='bottom')
plt.show()