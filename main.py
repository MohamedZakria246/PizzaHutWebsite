import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Read the dataset
data_df = pd.read_csv("CustomersDataset.csv")


# Get overview of the data
def dataoveriew(df, message):
    print(f'{message}:\n')
    print('Number of rows: ', df.shape[0])
    print("\nNumber of features:", df.shape[1])
    print("\nData Features:")
    print(df.columns.tolist())
    print("\nMissing values:", df.isnull().sum().values.sum())
    print("\nUnique values:")
    print(df.nunique())


dataoveriew(data_df, 'Overview of the dataset')

# feature scaling preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder


data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
data_df['TotalCharges'] = data_df['TotalCharges'].astype("float")

sc = StandardScaler()
data_df['tenure'] = sc.fit_transform(data_df['tenure'].values.reshape(-1, 1))
data_df['MonthlyCharges'] = sc.fit_transform(data_df['MonthlyCharges'].values.reshape(-1, 1))
data_df['TotalCharges'] = sc.fit_transform(data_df['TotalCharges'].values.reshape(-1, 1))
# Create a model
# Import metric for performance evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cleaned_data_df = data_df.drop(['customerID'], axis=1)

# convert all non numeric columns to numeric columns
for i in cleaned_data_df.columns:
    if cleaned_data_df[i].dtype == "+int64":
        continue
    else:
        cleaned_data_df[i] = LabelEncoder().fit_transform(cleaned_data_df[i])

data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'],errors='coerce')
data_df['TotalCharges'] = data_df['TotalCharges'].astype("float")

print(data_df[data_df['TotalCharges'] == ' '])

print(data_df.dtypes)
# pridect

genderLe = LabelEncoder()
data_df["gender"] = genderLe.fit_transform(data_df["gender"])
# print(le.inverse_transform([0]), "TEST#################################")

partnerLe = LabelEncoder()
data_df["Partner"] = partnerLe.fit_transform(data_df["Partner"])
# print(partenerLe.inverse_transform([1]), "TEST#************************************")

DependentsLe = LabelEncoder()
data_df["Dependents"] = DependentsLe.fit_transform(data_df["Dependents"])

PhoneServiceLe = LabelEncoder()
data_df["PhoneService"] = PhoneServiceLe.fit_transform(data_df["PhoneService"])

MultipleLinesLe = LabelEncoder()
data_df["MultipleLines"] = MultipleLinesLe.fit_transform(data_df["MultipleLines"])

InternetServiceLe = LabelEncoder()
data_df["InternetService"] = InternetServiceLe.fit_transform(data_df["InternetService"])

OnlineSecurityLe = LabelEncoder()
data_df["OnlineSecurity"] = OnlineSecurityLe.fit_transform(data_df["OnlineSecurity"])

OnlineBackupLe = LabelEncoder()
data_df["OnlineBackup"] = OnlineBackupLe.fit_transform(data_df["OnlineBackup"])

DeviceProtectionLe = LabelEncoder()
data_df["DeviceProtection"] = DeviceProtectionLe.fit_transform(data_df["DeviceProtection"])

TechSupportLe = LabelEncoder()
data_df["TechSupport"] = TechSupportLe.fit_transform(data_df["TechSupport"])

StreamingTVLe = LabelEncoder()
data_df["StreamingTV"] = StreamingTVLe.fit_transform(data_df["StreamingTV"])

StreamingMoviesLe = LabelEncoder()
data_df["StreamingMovies"] = StreamingMoviesLe.fit_transform(data_df["StreamingMovies"])

ContractLe = LabelEncoder()
data_df["Contract"] = ContractLe.fit_transform(data_df["Contract"])

PaperlessBillingLe = LabelEncoder()
data_df["PaperlessBilling"] = PaperlessBillingLe.fit_transform(data_df["PaperlessBilling"])

PaymentMethodLe = LabelEncoder()
data_df["PaymentMethod"] = PaymentMethodLe.fit_transform(data_df["PaymentMethod"])

data_df.TotalCharges = pd.to_numeric(data_df["TotalCharges"], errors='coerce', downcast='float')




def pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):


    Data = [(genderLe.transform([Gender]), SeniorCitizen, (partnerLe.transform([Partner])),
              DependentsLe.transform([Dependents]),
              tenure, PhoneServiceLe.transform([PhoneService]), MultipleLinesLe.transform([MultipleLines]),
              InternetServiceLe.transform([InternetService]),
              OnlineSecurityLe.transform([OnlineSecurity]), OnlineBackupLe.transform([OnlineBackup]),
              DeviceProtectionLe.transform([DeviceProtection]), TechSupportLe.transform([TechSupport]),
              StreamingTVLe.transform([StreamingTV]), StreamingMoviesLe.transform([StreamingMovies]),
              ContractLe.transform([Contract]), PaperlessBillingLe.transform([PaperlessBilling]),
              PaymentMethodLe.transform([PaymentMethod]), MonthlyCharges, TotalCharges)]


    df = pd.DataFrame(Data)

    return df


# Split data into train and test sets


from sklearn.model_selection import train_test_split

X = cleaned_data_df.drop('Churn', axis=1)
y = cleaned_data_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
X_pred = log_reg.predict(X_train)
y_pred = log_reg.predict(X_test)


def logistic_test():
    print("")
    print("Logistic Regression test")
    global acc_logistic_test, pre_logistic_test, rec_logistic_test, f1_logistic_test
    acc_logistic_test = accuracy_score(y_test, y_pred)
    pre_logistic_test = precision_score(y_test, y_pred)
    rec_logistic_test = recall_score(y_test, y_pred)
    f1_logistic_test = f1_score(y_test, y_pred)
    logistic_ts = [acc_logistic_test, pre_logistic_test, rec_logistic_test, f1_logistic_test]


    print('Accuracy of Logistic Regression Classifier:', acc_logistic_test)
    print('Precision of Logistic Regression Classifier:', pre_logistic_test)
    print('Recall of Logistic Regression Classifier:', rec_logistic_test)
    print('F1 Score of Logistic Regression Classifier:', f1_logistic_test)
    return logistic_ts


def logistic_train():
    print("")
    print("Logistic Regression train ")
    acc_logistic = accuracy_score(y_train, X_pred)
    pre_logistic = precision_score(y_train, X_pred)
    rec_logistic = recall_score(y_train, X_pred)
    f1_logistic = f1_score(y_train, X_pred)

    print(acc_logistic)
    print(pre_logistic)
    print(rec_logistic)
    print(f1_logistic)

    logistic_tr = [acc_logistic, pre_logistic, rec_logistic, f1_logistic]
    return logistic_tr

#SeniorCitizen و tenure و MonthlyCharges

def logistic_pridect(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):

    logisticPri = pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges)
    print(logisticPri.dtypes)
    predLogistic = log_reg.predict(logisticPri)

    print("\nPredict Logistic Regression  :")
    if predLogistic[0] == 1:
        print("Yes")
    else:
        print("NO")
    return predLogistic


logistic_train()


logistic_test()

logistic_pridect('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')


# pri = pridectfun('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')
# pred = log_reg.predict(pri)
# print("\n Predict Logistic Regression :")
# print(pred)


#-----------------------------------------------------------


# print(a, b)
print("\nSVM")
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predd = svc_model.predict(X_test)
X_predd = svc_model.predict(X_train)


# def svm ():


def svm_test():
    acc_svm_test = accuracy_score(y_test, y_predd)
    pre_svm_test = precision_score(y_test, y_predd)
    rec_svm_test = recall_score(y_test, y_predd)
    f1_svm_test = f1_score(y_test, y_predd)

    print("")
    print('Accuracy of svm Classifier:', acc_svm_test)
    print('Precision of svm Classifier:', pre_svm_test)
    print('Recall of svm Classifier:', rec_svm_test)
    print('F1 Score of svm Classifier:', f1_svm_test)

    svm_ts = [acc_svm_test, pre_svm_test, rec_svm_test, f1_svm_test]
    return svm_ts


def svm_train():
    acc_svm = accuracy_score(y_train, X_pred)
    pre_svm = precision_score(y_train, X_pred)
    rec_svm = recall_score(y_train, X_pred)
    f1_svm = f1_score(y_train, X_pred)


    print(acc_svm)
    print(pre_svm)
    print(rec_svm)
    print(f1_svm)

    svm_tr = [acc_svm, pre_svm, rec_svm, f1_svm]
    return svm_tr

def svm_pridect(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):

    svmPri = pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges)

    predsvm = svc_model.predict(svmPri)
    print("\nPredict svm :")
    if predsvm[0] == 1:
        print("Yes")
    else:
        print("NO")
    return predsvm




svm_train()

svm_test()

svm_pridect('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')


#-----------------------------------------------------------------


print("\nDecision tree")

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_of_dt = dt_model.predict(X_test)
X_pred_of_dt = dt_model.predict(X_train)


def Decision_test():
    acc_Decision_test = accuracy_score(y_test, y_pred_of_dt)
    pre_Decision_test = precision_score(y_test, y_pred_of_dt)
    rec_Decision_test = recall_score(y_test, y_pred_of_dt)
    f1_Decision_test = f1_score(y_test, y_pred_of_dt)


    print('Accuracy of Decision tree Classifier:', acc_Decision_test)
    print('Precision of Decision tree Classifier:', pre_Decision_test)
    print('Recall of Decision tree Classifier:', rec_Decision_test)
    print('F1 Score of Decision tree Classifier:', f1_Decision_test)

    id3_ts = [acc_Decision_test, pre_Decision_test, rec_Decision_test, f1_Decision_test]
    return id3_ts


def Decision_train():
    acc_Decision_train = accuracy_score(y_train, X_pred_of_dt)
    pre_Decision_train = precision_score(y_train, X_pred_of_dt)
    rec_Decision_train = recall_score(y_train, X_pred_of_dt)
    f1_Decision_tarin = f1_score(y_train, X_pred_of_dt)

    print("")
    print(acc_Decision_train)
    print(pre_Decision_train)
    print(rec_Decision_train)
    print(f1_Decision_tarin)

    id3_tr = [acc_Decision_train, pre_Decision_train, rec_Decision_train, f1_Decision_tarin]
    return id3_tr



def DecicionTree_pridect(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):

    DecicsionPri = pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges)

    predDecision = dt_model.predict(DecicsionPri)
    print("\nPredict Decision Tree :")
    if predDecision[0] == 1:
        print("Yes")
    else:
        print("NO")
    return predDecision


Decision_test()

Decision_train()

DecicionTree_pridect('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')

#------------------------------------------------------------------


print("\nNaive bayes")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_of_nb = nb_model.predict(X_test)
X_pred_of_nb = nb_model.predict(X_train)


def Naive_test():
    acc_Naive_test = accuracy_score(y_test, y_pred_of_nb)
    pre_Naive_test = precision_score(y_test, y_pred_of_nb)
    rec_Naive_test = recall_score(y_test, y_pred_of_nb)
    f1_Naive_test = f1_score(y_test, y_pred_of_dt)


    print('Accuracy of Naive bayes Classifier:', acc_Naive_test)
    print('Precision of Naive bayes Classifier:', pre_Naive_test)
    print('Recall of Naive bayes Classifier:', rec_Naive_test)
    print('F1 Score of Naive bayes Classifier:', f1_Naive_test)

    naive_ts = [acc_Naive_test, pre_Naive_test, rec_Naive_test, f1_Naive_test]
    return naive_ts


def Naive_train():
    acc_Naive_train = accuracy_score(y_train, X_pred_of_nb)
    pre_Naive_train = precision_score(y_train, X_pred_of_nb)
    rec_Naive_train = recall_score(y_train, X_pred_of_nb)
    f1_Naive_train = f1_score(y_train, X_pred_of_dt)

    print("")
    print(acc_Naive_train)
    print(pre_Naive_train)
    print(rec_Naive_train)
    print(f1_Naive_train)

    naive_tr = [acc_Naive_train, pre_Naive_train, rec_Naive_train, f1_Naive_train]
    return naive_tr


def Naive_pridect(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):

    NaivePri = pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges)

    predNaive = nb_model.predict(NaivePri)
    print("\nPredict Naive bayes :")
    if predNaive[0] == 1:
        print("Yes")
    else:
        print("NO")

    return predNaive


Naive_test()

Naive_train()

Naive_pridect('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')

#-----------------------------------------------------------

print("\nRandom Forest Classifier")
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
y_pred_of_RF = RF_model.predict(X_test)
X_pred_of_RF = RF_model.predict(X_train)


def Random_Test():
    acc_random_test = accuracy_score(y_test, y_pred_of_RF)
    pre_random_test = precision_score(y_test, y_pred_of_RF)
    rec_random_test = recall_score(y_test, y_pred_of_RF)
    f1_random_test = f1_score(y_test, y_pred_of_RF)

    print("")
    print('Accuracy of Random forest Classifier:', acc_random_test)
    print('Precision of Random forest Classifier:', pre_random_test)
    print('Recall of Random forest Classifier:', rec_random_test)
    print('F1 Score of Random forest Classifier:', f1_random_test)

    random_ts = [acc_random_test, pre_random_test, rec_random_test, f1_random_test]
    return random_ts


def Random_Train():
    acc_random_train = accuracy_score(y_train, X_pred_of_RF)
    pre_random_train = precision_score(y_train, X_pred_of_RF)
    rec_random_train = recall_score(y_train, X_pred_of_RF)
    f1_random_train = f1_score(y_train, X_pred_of_RF)


    print(acc_random_train)
    print(pre_random_train)
    print(rec_random_train)
    print(f1_random_train)

    random_tr = [acc_random_train, pre_random_train, rec_random_train, f1_random_train]
    return random_tr

def Random_pridect(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges):

    RandomPri = pridectfun(Gender, SeniorCitizen, Partner, Dependents,
               tenure, PhoneService, MultipleLines, InternetService,
               OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
               StreamingTV, StreamingMovies, Contract, PaperlessBilling,
               PaymentMethod, MonthlyCharges, TotalCharges)

    predRandom = RF_model.predict(RandomPri)
    print("\nPredict Random Forest Classifier :")
    if predRandom[0] == 1:
        print("Yes")
    else:
        print("NO")

    return predRandom


Random_Train()

Random_Test()

Random_pridect('Female', 0, 'Yes', 'No', 1, 'Yes', 'No phone service', 'DSL', 'No', 'No', 'No', 'No', 'No', 'Yes', 'Month-to-month', 'Yes', 'Electronic check', 29.85, '29.85')


