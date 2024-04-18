#Importing the required Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sns.set()
warnings.simplefilter('ignore')

#Data Preprocessing
data = pd.read_csv('creditcard.csv')
df = data.copy() # To keep the data as backup
df.head()

df.shape

df.isnull().sum()

df.isnull().values.any()

df.dtypes

df.Time.tail(15)

df.describe()

#Checking the frequency of frauds before moving forward
df.Class.value_counts()

sns.countplot(x=df.Class, hue=df.Class)

#Checking the distribution of amount
plt.figure(figsize=(10, 5))
sns.distplot(df.Amount)


#Since, it is a little difficult to see. Let's engineer a new feature of bins.
df['Amount-Bins'] = ''

#Now, let's set the bins and their labels.
def make_bins(predictor, size=50):
    '''
    Takes the predictor (a series or a dataframe of single predictor) and size of bins
    Returns bins and bin labels
    '''
    bins = np.linspace(predictor.min(), predictor.max(), num=size)

    bin_labels = []

    # Index of the final element in bins list
    bins_last_index = bins.shape[0] - 1

    for id, val in enumerate(bins):
        if id == bins_last_index:
            continue
        val_to_put = str(int(bins[id])) + ' to ' + str(int(bins[id + 1]))
        bin_labels.append(val_to_put)
    
    return bins, bin_labels


bins, bin_labels = make_bins(df.Amount, size=10)


#Now, adding bins in the column Amount-Bins.
df['Amount-Bins'] = pd.cut(df.Amount, bins=bins,
                           labels=bin_labels, include_lowest=True)
df['Amount-Bins'].head().to_frame()


#Let's plot the bins.
df['Amount-Bins'].value_counts()

plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df)
plt.xticks(rotation=45)


# Since, count of values of Bins other than '0 to 2854' are difficult to view. Let's not insert the first one.
plt.figure(figsize=(15, 10))
sns.countplot(x='Amount-Bins', data=df[~(df['Amount-Bins'] == '0 to 2854')])
plt.xticks(rotation=45)

#Predictive Modelling
#One-hot encoding the Amount-Bins

df_encoded = pd.get_dummies(data=df, columns=['Amount-Bins'])
df = df_encoded.copy()


df.head()

#Breaking the dataset into training and testing
X = df.drop(labels='Class', axis=1)
Y = df['Class']

X.shape, Y.shape

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, random_state=42, test_size=0.3, shuffle=True)

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()

# Training the algorithm
lr_model.fit(xtrain, ytrain)


# Predictions on training and testing data
lr_pred_train = lr_model.predict(xtrain)
lr_pred_test = lr_model.predict(xtest)


# Importing the required metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

tn, fp, fn, tp = confusion_matrix(ytest, lr_pred_test).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


sns.heatmap(conf_matrix, annot=True)

#accuracy score.
lr_accuracy = accuracy_score(ytest, lr_pred_test)
lr_accuracy

#precision and recall
lr_precision = precision_score(ytest, lr_pred_test)
lr_precision

lr_recall = recall_score(ytest, lr_pred_test)
lr_recall

#recall for training dataset to get the idea of any overfitting we may be having
lr_recall_train = recall_score(ytrain, lr_pred_train)
lr_recall_train

#F1-Score. F1-Score may tell us that one of the precision or recall is very low
from sklearn.metrics import f1_score
lr_f1 = f1_score(ytest, lr_pred_test)
lr_f1

#classification report
from sklearn.metrics import classification_report

print(classification_report(ytest, lr_pred_test))

#Now, for the ROC Curve, we need the probabilites of Fraud happening (which is the probability of occurance of 1)
lr_pred_test_prob = lr_model.predict_proba(xtest)[:, 1]

#Now, to draw the ROC Curve, we need to have True Positive Rate and False Positive Rate
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(ytest, lr_pred_test_prob)

#Also, let's get the auc score.
lr_auc = roc_auc_score(ytest, lr_pred_test_prob)
lr_auc

#Now, let's define a function to plot the roc curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8, 6))
    plt.title('ROC Curve', fontsize=15)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel('False Positive Rates', fontsize=15)
    plt.ylabel('True Positive Rates', fontsize=15)
    plt.legend(loc='best')
    
    plt.show()
    
 
#Plotting ROC Curve
plot_roc_curve(fpr=fpr, tpr=tpr, label="AUC = %.3f" % lr_auc)

#Model Complexity
#Let's try to train the Logistic Regression models on the 2nd degree of polynomials. 
#Not going further 2nd degree because features are already too much. Otherwise, computer gives the MemoryError.
from sklearn.preprocessing import PolynomialFeatures

# Getting the polynomial features
poly = PolynomialFeatures(degree=2)
xtrain_poly = poly.fit_transform(xtrain)
xtest_poly = poly.fit_transform(xtest)

# Training the model
model = LogisticRegression()
model.fit(xtrain_poly, ytrain)

# Getting the probabilities
train_prob = model.predict_proba(xtrain_poly)[:, 1]
test_prob = model.predict_proba(xtest_poly)[:, 1]

# Computing the ROC Score
roc_auc_score(ytrain, train_prob), roc_auc_score(ytest, test_prob)  

#Plotting ROC Curve for the Test data.
fpr_poly, tpr_poly, threshold_poly = roc_curve(ytest, test_prob)

plot_roc_curve(fpr=fpr_poly, tpr=tpr_poly, label='AUC = %.3f' %  roc_auc_score(ytest, test_prob)) 

#Let's also check the Recall in case of model complexity
recall_score(ytest, model.predict(xtest_poly)) 



#Support Vector Machine

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# Let's first check the head of the explanatory variables which are to be scaled.
X.head()

X_scaled = mms.fit_transform(X)


X_scaled = pd.DataFrame(data=X_scaled, columns=X.columns)
X_scaled.head()


#Now, let's train test split on the scaled data

xtrainS, xtestS, ytrainS, ytestS = train_test_split(
    X_scaled, Y, random_state=42, test_size=0.30, shuffle=True)


print(xtrainS.shape, ytrainS.shape)
print(xtestS.shape, ytestS.shape)


from sklearn.svm import SVC

svc_model = SVC(kernel='linear', probability=True)


svc_model.fit(xtrainS, ytrainS)


svc_pred = svc_model.predict(xtestS)


svc_recall = recall_score(ytestS, svc_pred)


svc_recall


#Recall quite increased in case of SVC
svc_pred_prob = svc_model.predict_proba(xtestS)[:, 1]

#Now, let's draw the ROC Curve.
# First, getting the auc score
svc_auc = roc_auc_score(ytestS, svc_pred_prob)

# Now, let's get the fpr and tpr
fpr, tpr, threshold = roc_curve(ytestS, svc_pred_prob)

# Now, let's draw the curve
plot_roc_curve(fpr, tpr, 'AUC: %.3f' % svc_auc)


#Tuning the Hyper-parameters
# For Kernel = rbf
tuned_rbf = {'kernel': ['rbf'], 'gamma': [
    1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}

# For kernel = sigmoid
tuned_sigmoid = {'kernel': ['sigmoid'], 'gamma': [
    1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}

# For kernel = linear
tuned_linear = {'kernel': ['linear'], 'C': [
    0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}


from sklearn.model_selection import RandomizedSearchCV

rs_rbf = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_rbf, n_iter=500, n_jobs=4, scoring='roc_auc')

rs_sigmoid = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_sigmoid, n_iter=500, n_jobs=4, scoring='roc_auc')

rs_linear = RandomizedSearchCV(estimator=SVC(probability=True), 
        param_distributions=tuned_linear, n_iter=500, n_jobs=4, scoring='roc_auc')


#For kernel rbf:


rs_rbf.fit(xtrainS, ytrainS)

rs_rbf.best_estimator_
svc_rbf_best_est = rs_rbf.best_estimator_
svc_rbf_best_est.fit(xtrainS, ytrainS)

svc_rbf_best_est_pred = svc_rbf_best_est.predict(xtestS)

svc_rbf_best_est_pred_proba = svc_rbf_best_est.predict_proba(xtestS)[:, 1]

#Getting the AUC Score
svc_rbf_auc = roc_auc_score(ytestS, svc_rbf_best_est_pred_proba)

#Getting the Recall
svc_rbf_recall = recall_score(ytestS, svc_rbf_best_est_pred)
svc_rbf_recall


#We can see that in this model, both recall and ROC Score are great. Let's draw the ROC Curve.

fpr, tpr, threshold = roc_curve(ytestS, svc_rbf_best_est_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_rbf_auc)


#Now, for kernel sigmoid.
rs_sigmoid.fit(xtrainS, ytrainS)

svc_sigmoid = rs_sigmoid.best_estimator_

svc_sigmoid.fit(xtrainS, ytrainS)

svc_sigmoid_pred = svc_sigmoid.predict(xtestS)
svc_sigmoid_pred_proba = svc_sigmoid.predict_proba(xtestS)[:, 1]

#AUC:

svc_sigmoid_auc = roc_auc_score(ytestS, svc_sigmoid_pred_proba)
svc_sigmoid_auc



#Recall:

svc_sigmoid_recall = recall_score(ytestS, svc_sigmoid_pred)
svc_sigmoid_recall

fpr, tpr, threshold = roc_curve(ytestS, svc_sigmoid_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_sigmoid_auc)


#Let's check for Linear kernel.
rs_linear.fit(xtrainS, ytrainS)

svc_linear = rs_linear.best_estimator_

svc_linear.fit(xtrainS, ytrainS)

svc_linear_pred = svc_linear.predict(xtestS)
svc_linear_pred_proba = svc_linear.predict_proba(xtestS)[:, 1]

#AUC and ROC Curve

svc_linear_auc = roc_auc_score(ytestS, svc_linear_pred_proba)

fpr, tpr, threshold = roc_curve(ytestS, svc_linear_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % svc_linear_auc)

svc_linear_recall = recall_score(ytestS, svc_linear_pred)
svc_linear_recall


#Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()


nb.fit(xtrain, ytrain)

nb_pred = nb.predict(xtest)
nb_pred_proba = nb.predict_proba(xtest)[:, 1]


nb_auc = roc_auc_score(ytest, nb_pred)

fpr, tpr, threshold = roc_curve(ytestS, nb_pred_proba)
plot_roc_curve(fpr, tpr, 'AUC = %.3f' % nb_auc)


nb_recall = recall_score(ytest, nb_pred)
nb_recall

#Conclusion: Naive Bayes didn't perform well as compared to the other ones.

