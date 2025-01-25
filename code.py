import pandas as p
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_ind
from scipy.stats import  kstest
import numpy as np

#Metrics
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

"""
This Code process the data about alzheimer patients.
the data is taken from this link: 
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

In summary, this dataset contains extensive health information for 2,149 patients.
The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, 
cognitive and functional assessments, symptoms, and a diagnosis of Alzheimer's Disease. 
The data is ideal for researchers and data scientists looking to explore factors associated 
with Alzheimer's, develop predictive models, and conduct statistical analyses.
The labels are already encoded.

the columns are: 'PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
       'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
       'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
       'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP',
       'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
       'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
       'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion',
       'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
       'Forgetfulness', 'Diagnosis', 'DoctorInCharge'

The operations are the following: 
- cleaning the data.
- Show ethnicity diversity of the patients. 
- Show gender diversity of the patients. 
- Show education diversity of the patients. 
- Show ages of the patients. 
- Show ages of the patients.
- The distribution of the Lifestyle Factors.
- The distribution of the Clinical Measurements.
- Display the correlation of the numerical value.
- Relation of the Diet Quality with Depression.
- Relation of the Education level with MMSE.
- Creation of Bays model for predicting Memory Complaints using Lifestyle Factors.
- Creation of Decision tree model for predicting Memory Complaints using Lifestyle Factors.
"""


"""
This function takes a camel case string and converts it to a sentence
"""
def camelToSentence(camel_str):
    # Add space before each uppercase letter, except at the start
    sentence = re.sub(r'(?<!^)(?=[A-Z])', ' ', camel_str)
    # Capitalize the first letter of the sentence
    return sentence.capitalize()

""" 
this function to clean the data.
"""
def cleanData(mdata):
    # make a copy of the data, to avoid manipulation to the original data.
    newData = mdata.copy()
    # replace the numerical values in the columns with the actual descriptive values
    newData["Ethnicity"] = newData['Ethnicity'].replace({0: "Caucasian",
                                                         1: "African American",
                                                         2:"Asian",
                                                         3:"Other"})
    newData["Gender"] = newData['Gender'].replace({0: "Male",
                                                   1: "Female"})
    newData["EducationLevel"] = newData['EducationLevel'].replace({0: "None",
                                                                   1: "High School",
                                                                   2:"Bachelor's",
                                                                   3:"Higher Education"})
    # dropping the column DoctorInCharge as it is not needed for the analysis and only contains the same value
    newData =newData.drop(columns=['DoctorInCharge',"PatientID"])
    return newData

def displayEthnicityData(mdata):
    x=["Caucasian", "African American","Asian","Other"]
    y=[mdata.loc[mdata["Ethnicity"]==i].shape[0] for i in x]
    plt.bar(x,y,color='blue', edgecolor='navy')
    plt.title('Ethnicity Distribution')
    plt.xlabel('Ethnicity')
    plt.ylabel('Frequency')
    plt.savefig('./images/Ethnicity Distribution.png') 
    plt.show()
    
    
def displayGenderData(mdata):
    x=["Male", "Female"]
    # y=[mdata.loc[mdata["Gender"]==i].shape[0] for i in x]
    y=[mdata.loc[mdata["Gender"]==i].shape[0] for i in x]
    # plt.bar(x,y,color='blue', edgecolor='navy')
    plt.pie(y, labels=x, colors=['blue', 'red'], autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.savefig('./images/Gender Distribution.png')
    plt.show()  

def displayEducationLevelData(mdata):
    x=["None", "High School","Bachelor's", "Higher Education"]
    y=[mdata.loc[mdata["EducationLevel"]==i].shape[0] for i in x]
    plt.bar(x,y,color='blue', edgecolor='navy')
    plt.title('Education Level Distribution')
    plt.xlabel('Education Level')
    plt.ylabel('Frequency')
    plt.savefig('./images/Education Level.png')
    plt.show()   
    
def displayAgeData(mdata):
    plt.hist(mdata.loc[:,"Age"],bins=30,color='blue', edgecolor='navy')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('./images/Age Distribution.png')   
    plt.show()
    
def displayBoxPlotForLifestyleFactorsData(mdata):
    x=mdata.loc[:,"BMI":"SleepQuality"].drop(columns=['Smoking'])
    plt.boxplot(x)
    plt.title('Lifestyle Factors Plots', fontsize=16)
    plt.ylabel('Values (Objective Rating)', fontsize=12)
    plt.xticks([i+1 for i in range(mdata.columns[6:11].shape[0])],
               [camelToSentence(name) for name in mdata.columns[6:11]], rotation=0, fontsize=6)
    plt.savefig('./images/Lifestyle Factors.png')
    plt.show()    
    
def displayBoxPlotForClinicalMeasurementsData(mdata):
    x=mdata.loc[:,"SystolicBP":"CholesterolTriglycerides"]
    plt.boxplot(x)
    plt.title('Clinical Measurements Plots', fontsize=16)
    plt.ylabel('Values (mg/dL)', fontsize=12)
    plt.xticks([i+1 for i in range(mdata.columns[16:22].shape[0])],
               [camelToSentence(name) for name in mdata.columns[16:22]], rotation=0, fontsize=6)
    plt.savefig('./images/Clinical Measurements.png')
    plt.show()     

def displayCorrelation(mdata):
    # delete encoded values.
    newData =mdata.drop(columns=
                        ['Gender', 'Ethnicity', 'EducationLevel',
                        'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                        'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
                        'MemoryComplaints', 'BehavioralProblems', 'ADL',
                        'Confusion','Disorientation', 'PersonalityChanges', 
                        'DifficultyCompletingTasks','Forgetfulness',])
    plt.matshow(newData.corr())  
    plt.title('Correlation Between Data', fontsize=16)
    plt.savefig('./images/Correlation Between Data.png')  
    

"""
This function takes the data and calculates the Kolmogorov-Smirnov Test for the Diet Quality 
based on Depression.
Does the Depression  affect the Diet Quality for people in the 60 to 90 age group?.
the sample size is 30 for each group.
H0: Diet Quality is the same for people with and without depression.
H1: Diet Quality is different for people with and without depression.
"""
def dietQualityWithDepression(mdata):
    # Filter groups
    group_with_depression = mdata[mdata['Depression'] == 1]['DietQuality'].head(30)
    group_without_depression = mdata[mdata['Depression'] == 0]['DietQuality'].head(30)
    # Kolmogorov-Smirnov Test for group_with_depression 
    print("Kolmogorov-Smirnov Test for group_with_depression")
    ks_stat, ks_p = kstest(group_with_depression, 'norm', args=(group_with_depression.mean(), group_with_depression.std()))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, P-Value={ks_p}")
    if ks_p < 0.05:
        print("Reject the null hypothesis: Data is not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Data is normally distributed.")
    # Kolmogorov-Smirnov Test for group_without_depression 
    print("Kolmogorov-Smirnov Test for group_without_depression")
    ks_stat, ks_p = kstest(group_without_depression, 'norm', args=(group_without_depression.mean(), group_without_depression.std()))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, P-Value={ks_p}")
    if ks_p < 0.05:
        print("Reject the null hypothesis: Data is not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Data is normally distributed.")
    # Perform T-test
    print("T-Test for Diet Quality based on Depression")
    t_stat, p_value = ttest_ind(group_with_depression, group_without_depression)

    # Results
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: Diet Quality differs based on depression.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in Diet Quality.")
"""
This function takes the data and calculates the Kolmogorov-Smirnov Test for 
the MMSE Mini-Mental State Examination score based on Education Level.
Does the Education Level affect the MMSE for people in the 60 to 90 age group?.
the sample size is 30 for each group.
H0: MMSE is the same for people with and without higher education.
H1: MMSE is different for people with and without higher education.
"""        
def educationLevelWithMMSE(mdata):
    # Filter groups
    group_without_education  = mdata[(mdata['EducationLevel'] == "None" ) | (mdata['EducationLevel'] == "High School")]['MMSE'].head(30)
    group_with_education = mdata[(mdata['EducationLevel'] == "Bachelor's") | (mdata['EducationLevel'] == "Higher Education")]['MMSE'].head(30)
    # Kolmogorov-Smirnov Test for group_with_depression 
    print("Kolmogorov-Smirnov Test for group_with_education")
    ks_stat, ks_p = kstest(group_with_education, 'norm', args=(group_with_education.mean(), group_with_education.std()))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, P-Value={ks_p}")
    if ks_p < 0.05:
        print("Reject the null hypothesis: Data is not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Data is normally distributed.")
    # Kolmogorov-Smirnov Test for group_without_education 
    print("Kolmogorov-Smirnov Test for group_without_education")
    ks_stat, ks_p = kstest(group_without_education, 'norm', args=(group_without_education.mean(), group_without_education.std()))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, P-Value={ks_p}")
    if ks_p < 0.05:
        print("Reject the null hypothesis: Data is not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Data is normally distributed.")
    # Perform T-test
    print("T-Test for MMSE based on Education Level")
    t_stat, p_value = ttest_ind(group_with_education, group_without_education)

    # Results
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
    if p_value < 0.05:
        print("Reject the null hypothesis: MMSE differs based on Education Level.")
    else:
        print("Fail to reject the null hypothesis: No significant difference in MMSE.")
"""
This function takes the data and creates a model for predicting Memory Complaints using Lifestyle Factors.
The model uses the Naive Bayes Classifier.
"""
def memoryComplaintsModelWithBayesClassifier(mdata):
    x = mdata.loc[:, 'BMI':'SleepQuality'].values
    y = mdata.loc[:, 'MemoryComplaints'].values
    accuracy_score = []
    max_accuracy = 0
    max_ratio=0
    model=GaussianNB()
    for i in range(2,18):
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=i*0.05,random_state=0)
        #Gaussian Naive Bayes
        #Classification algorithm for binary and
        #multi-class classification problems.
        gaussian = GaussianNB()
        gaussian.fit(X_train, y_train)
        Y_pred = gaussian.predict(X_test)
        accuracy_nb=round(metrics.accuracy_score(y_test,Y_pred)* 100, 2)
        acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
        #Model evaluation
        accuracy = metrics.accuracy_score(y_test,Y_pred)
        accuracy_score.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            model = gaussian
            max_ratio=i*0.05
    print(f'The accuracy of the Naive Bayes is {max_accuracy} at {max_ratio}')
    plt.plot([x*0.05 for x in range(2,18)],accuracy_score)
    plt.xlabel('Test Size')
    plt.ylabel('Bayes Classifier Accuracy')
    # plt.savefig('./images/Bayes Classifier Accuracy.png') 
    plt.show()
    return model
   
"""
This function takes the data and creates a model for predicting Memory Complaints using Lifestyle Factors.
The model uses the Decision Tree Classifier.
"""    
def memoryComplaintsModelWithDecisionTreeClassifier(mdata):
    x = mdata.loc[:, 'BMI':'SleepQuality'].values
    y = mdata.loc[:, 'MemoryComplaints'].values
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    accuracy_score = []
    max_accuracy = 0
    max_depth=0
    model=DecisionTreeClassifier()
    for i in range(1,10):
        #Decision tree
        #Create the tree
        mod_dt = DecisionTreeClassifier(max_depth = i, random_state = 1)
        #Train the model
        mod_dt.fit(X_train,y_train)
        #Test the model
        prediction=mod_dt.predict(X_test)
        #Print accuracy
        accuracy=metrics.accuracy_score(prediction,y_test)
        # print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy))
        accuracy_score.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            model = mod_dt
            max_depth = i
        #Print the importance of each predictor
        # print(mod_dt.feature_importances_)
    print(f'The accuracy of the Decision Tree is {max_accuracy} at {max_depth}')
    plt.plot(range(1,10),accuracy_score)
    plt.xlabel('Max Depth')
    plt.ylabel('Decision Tree Accuracy')
    plt.savefig('./images/Decision Tree Accuracy.png')
    plt.show() 
    return model

# model: the model to be used for prediction    
# BMI: Body Mass Index of the patients, ranging from 15 to 40.
# Smoking: Smoking status, where 0 indicates No and 1 indicates Yes.
# AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20.
# PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10.
# DietQuality: Diet quality score, ranging from 0 to 10.
# SleepQuality: Sleep quality score, ranging from 4 to 10.
def predictMemoryComplaints(model, BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality):
    pred = model.predict([[BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality]])
    print(pred)
    return pred[0]


data = p.read_csv('data.csv')

cleanedData = cleanData(data)
# displayEthnicityData(cleanedData)
# displayGenderData(cleanedData)
# displayEducationLevelData(cleanedData)
# displayAgeData(cleanedData)
# displayBoxPlotForLifestyleFactorsData(cleanedData)
# displayBoxPlotForClinicalMeasurementsData(cleanedData)
# displayCorrelation(cleanedData)
# dietQualityWithDepression(cleanedData)
# educationLevelWithMMSE(cleanedData)
# model = memoryComplaintsModelWithBayesClassifier(cleanedData)
# prediction = predictMemoryComplaints(model, 25, 7, 5, 1, 1, 7)
# if prediction == 0:
#     print("The patient is not likely to have memory complaints.")
# else:
#     print("The patient is likely to have memory complaints.")
model = memoryComplaintsModelWithDecisionTreeClassifier(cleanedData)
prediction = predictMemoryComplaints(model, 25, 7, 5, 1, 1, 7)
if prediction == 0:
    print("The patient is not likely to have memory complaints.")
else:
    print("The patient is likely to have memory complaints.")
# print(cleanedData.head())