#!/usr/bin/python
# -*- coding: cp1252 -*-

### IMPORTS ###
import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
sys.path.append("../tools/")

from helper_functions import *
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from pprint import pprint 


def main():
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    financial_features = ['salary', 'deferral_payments', 'total_payments', \
                         'loan_advances', 'bonus', 'restricted_stock_deferred',\
                         'deferred_income', 'total_stock_value', 'expenses', \
                         'exercised_stock_options', 'other', 'long_term_incentive', \
                         'restricted_stock', 'director_fees'] #(all units are in US dollars)

    email_features = ['to_messages', 'from_poi_to_this_person',
                     'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
    #(units are generally number of emails messages; notable exception is ‘email_address’, 
    # which is a text string)
    #email_address feature was removed from list

    poi_label = ['poi'] ###(boolean, represented as integer)

    features_list = poi_label + email_features + financial_features

    ### Load the dictionary containing the dataset
    with open("final_project_dataset_unix.pkl", "rb") as data_file:
        data_dict = pickle.load(data_file)
      
    #convert to a pandas dataframe for exploratory analysis
    df = pd.DataFrame.from_dict(data_dict, orient='index')

    #iterate df and convert string 'NaN' to actual np.nan
    for label, content in df.items():
        if label == 'email_address':
            for i in content:
                if i == 'NaN':
                    df[label][i] = np.nan
        else:
            df[label] = pd.to_numeric(df[label], errors='coerce')


    ### Investigate contents of dataset:
            
    # Total Number of data points
    total_people = df.shape[0]
    print('The total number of data points (people) in our data set is {}.\n'\
        .format(total_people))

    # Total Number of Features Used
    all_features = df.shape[1]
    print('There are {} features for each person in our dataset.\n'\
        .format(all_features))

    # Total Number of Persons Of Interest (POIs)
    poi_count = df['poi'][(df['poi'] == True)].count()
    print('Our dataset has {} persons of interest.\n'.format(poi_count))

    # Total Number of Non-POIs
    non_poi_count = total_people - poi_count
    print('Our dataset has {} Non persons of interest.\n'.format(non_poi_count))

    # Features with missing values?
    print('The following categories have missing values (NaN values)\n')
    print (df.isna().sum())


    ### Task 2: Remove outliers

    #visualize_features('salary', 'bonus', data_dict)
    #visualize_features('from_poi_to_this_person', 'from_this_person_to_poi', data_dict)
    #visualize_features('loan_advances', 'total_stock_value', data_dict)


    print()
    print('Searching for Outliers...')
    find_outlier('salary', df)
    print ()
    find_outlier('bonus', df)
    print()
    find_outlier('from_poi_to_this_person', df)
    print ()
    find_outlier('from_this_person_to_poi', df)
    print ()
    find_outlier('loan_advances', df)
    print ()
    find_outlier('total_stock_value', df)


    #get a count of number of NaN columns for each person
    nan_count = df.isna().sum(axis=1)


    print('\nThe top 5 people by number of NaN columns are:\n')
    print (nan_count.sort_values(ascending=False).head(5))

    print('\nLooking closer at Eugene Lockhart...\n')
    print( df.loc['LOCKHART EUGENE E'])

    print ('\nLooking closer at THE TRAVEL AGENCY IN THE PARK...\n')
    print (df.loc['THE TRAVEL AGENCY IN THE PARK'])


    ### Remove outliers
    df = df.drop(['TOTAL'], axis=0)
    df = df.drop(["LOCKHART EUGENE E"], axis=0)
    df = df.drop(["THE TRAVEL AGENCY IN THE PARK"], axis=0)

    #replace NaN with 0
    df = df.fillna(0)


    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = df.to_dict('index')

    for person in my_dataset:
        to_poi_count = my_dataset[person]['from_this_person_to_poi']
        from_poi_count = my_dataset[person]['from_poi_to_this_person']
        total_received_emails = my_dataset[person]['from_messages']
        total_sent_emails = my_dataset[person]['to_messages']
        
        try:
            my_dataset[person]['to_poi_ratio'] = float(to_poi_count) /\
                float(total_sent_emails)
        except:
            my_dataset[person]['to_poi_ratio'] = 0
        try:
            my_dataset[person]['from_poi_ratio'] = float(from_poi_count) /\
            float(total_received_emails)
        except:
            my_dataset[person]['from_poi_ratio'] = 0

    features_list = features_list + ['to_poi_ratio', 'from_poi_ratio']

    ### Preprocessing

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #Scaling features (normalizing all features)
    min_max_scaler = MinMaxScaler()
    features = min_max_scaler.fit_transform(features)

    ### Select the best features:
    # Removes all but the k highest scoring features
    n = 6 # adjust for optimization
    skb = SelectKBest(f_classif, k=n)
    skb.fit_transform(features, labels)
    #pprint(sorted(skb.scores_, reverse=True))

    #skip poi feature and combine with returned scores (key:value --> feature:score)
    scores = zip(features_list[1:], skb.scores_)

    #sort by highest scoring feature from scores
    sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
    #print '\nOur {} highest feature scores are:'.format(n)
    #pprint(sorted_scores[:n])
                                          
    #add k highest scoring features to create new features_list
    new_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[:n]
    #print '\nOur new features list includes: '
    #pprint(new_features_list)

    ### Extract features and labels from dataset using optimized features_list
    data = featureFormat(my_dataset, new_features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)


    ### Task 4: Try a variety of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html


    print ('\nRunning GaussianNB classifier...')
    run_classifier(GaussianNB(), features, labels)

    print ('\nRunning SVM classifier...')
    run_classifier(SVC(), features, labels)

    print ('\nRunning AdaBoost classifier...')
    run_classifier(AdaBoostClassifier(), features, labels)

    print ('\nRunning DecisionTree classifier...')
    run_classifier(DecisionTreeClassifier(), features, labels)



    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    ### Re-Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)


    # Adjust SVM parameters to refine accuracy
    # variables will be passed to fine_tune_algorithm to use in a Pipeline
    print ('\nThe best fit SVM has the following scores:\n')
    svm_steps = [('scaler', MinMaxScaler()), ('SKB', SelectKBest()),
                 ('SVM', SVC())]
    svm_parameters = {'SVM__kernel': ('linear', 'rbf'), 
                  'SVM__C':[0.001, 0.01, .1, 1, 10, 100, 1000], 
                  'SVM__gamma':[0.01, .1, 1, 10, 100, 1000],
                     'SKB__k': [2,3,4,5,6,7,8,9,10]}
    svm_clf = fine_tune_algorithm(svm_steps, svm_parameters, features, labels)


    # Adjust DecisionTreeClassifier parameters to refine accuracy
    print ('\nThe best fit DecisionTreeClassifer has the following scores:\n')
    dt_steps = [('scaler', MinMaxScaler()), ('SKB', SelectKBest()), 
                ('DT', DecisionTreeClassifier())]
    dt_parameters = {'DT__criterion': ('gini', 'entropy'), 
                  'DT__min_samples_split':[2,3,4,5,6,7,8,9,10],
                     'DT__random_state':[13],
                     'SKB__k': [2,3,4,5,6,7,8,9,10]}
    dt_clf = fine_tune_algorithm(dt_steps, dt_parameters, features, labels)


    # Adjust AdaBoostClassifier parameters to refine accuracy
    # variables will be passed to fine_tune_algorithm to use in a Pipeline
    print ('\nThe best fit AdaBoostClassifier has the following scores:\n')
    ab_steps = [('scaler', MinMaxScaler()), ('SKB', SelectKBest()),
                ('AB', AdaBoostClassifier())]
    ab_parameters = {'AB__algorithm': ('SAMME', 'SAMME.R'), 
                  'AB__learning_rate':[.5, .6, .7, .8, .9,1],
                     'SKB__k': [2,3,4,5,6,7,8,9,10]}
    ada_clf = fine_tune_algorithm(ab_steps, ab_parameters, features, labels)

    # Adjust GaussianNB parameters to refine accuracy
    print ('\nThe best fit GaussianNB Classifier has the following scores:\n')
    nb_steps = [('scaler', MinMaxScaler()), ('SKB', SelectKBest()),
                ('NB', GaussianNB())]
    nb_parameters = {'SKB__k': [2,3,4,5,6,7,8,9,10]}
    nb_clf = fine_tune_algorithm(nb_steps, nb_parameters, features, labels)

    #final best fitting classifier
    clf = nb_clf

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.

    dump_classifier_and_data(clf, my_dataset, features_list)


if __name__ == '__main__':
    main()
