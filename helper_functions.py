
import sys
sys.path.append("../tools/")

from feature_format import featureFormat
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


def visualize_features(x_feature, y_feature, data_dict):
    ###This function will extract specified features and plot them as a scatter plot
    
    data = featureFormat(data_dict, [x_feature, y_feature])
    for point in data:
        x = point[0]
        y = point[1]
        plt.scatter(x,y)        
    plt.ylabel(y_feature)
    plt.xlabel(x_feature)
    plt.show()


def find_outlier(feature, df):
    #takes a specified feature and shows 4 people with highest values for said feature
    
    print('\nThe top 4 highest {} are: \n'.format(feature))
    temp_df = df.sort_values(feature,ascending=False, na_position='last').head(4)
    print (temp_df[feature])


def run_classifier(clf, features, labels, iterations = 100):
    #Function will execute a classifier, fit it, and predict accuracy, precision,
    # and recall scores (iterations is adjustable)
    
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i in range(iterations):
       # Split data into testing and training groups
        features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size=0.3,
                                 stratify = labels, random_state = i)
        
        clf.fit(features_train, labels_train) #fit classifier                          
        pred = clf.predict(features_test) #predict labels

        #calculate meaningful stats stored as a list
        accuracy_scores = accuracy_scores + [accuracy_score(labels_test, pred)]
        precision_scores = precision_scores + [precision_score(labels_test, pred)]
        recall_scores = recall_scores + [recall_score(labels_test, pred)]                    
        f1_scores = f1_scores + [f1_score(labels_test, pred)]

    print ('\nFor {} iterations of different features/labels splits, our results are:'\
               .format(iterations))
    print('Mean Accuracy Score is: {}'.format(stat.mean(accuracy_scores)))
    print( 'Median Accuracy Score is: {}\n'.format(stat.median(accuracy_scores)))
    
    if round(sum(precision_scores)) > 0:
        #to ensure we have samples that are not all 0s, which allows scores to be calculated
        #without getting an error
        
        print ('Mean Precision Score is: {}'.format(stat.mean(precision_scores)))
        print ('Median Precision Score is: {}\n'.format(stat.median(precision_scores)))
        print ('Mean Recall Score is: {}'.format(stat.mean(recall_scores)))
        print ('Median Recall Score is: {}\n'.format(stat.median(recall_scores)))
        print ('Mean f1 Score is: {}'.format(stat.mean(f1_scores)))
        print ('Median f1 Score is: {}'.format(stat.median(f1_scores)))
    else:
        print ('Mean Precision Score is: 0.0')
        print ('Median Precision Score is: 0.0\n')
        print ('Mean Recall Score is: 0.0')
        print ('Median Recall Score is: 0.0\n')
        print ('Mean f1 Score is: 0.0')
        print ('Median f1 Score is: 0.0')


def fine_tune_algorithm(steps, parameters, features, labels):
    #function will initialize pipeline using passed in parameters      

    # Split data into static testing and training groups
    features_train, features_test, labels_train, labels_test = \
                train_test_split(features, labels, test_size=0.3,
                                 shuffle=True,
                                 stratify = labels, random_state = 13)
    
    pipeline = Pipeline(steps) #make a Pipeline object
   
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    #create a GridSearchCV classifier
    clf = GridSearchCV(pipeline, param_grid = parameters,
                       cv = sss, n_jobs = -1, scoring='f1', iid=False)
    
    clf.fit(features_train, labels_train) #fit classifier                          
    pred = clf.predict(features_test) #predict labels

    print ('Accuracy Score: {}'.format(accuracy_score(labels_test, pred)))
    print ('Precision Score: {}'.format(precision_score(labels_test, pred)))
    print ('Recall Score: {}'.format(recall_score(labels_test, pred)))
    print ('F1 Score: {}'.format(f1_score(labels_test, pred)))
    
    print ('\nThe best fit parameters are:')
    best_params = clf.best_params_
    for key, value in best_params.items():
        print ('{} : {}'.format(key, value))

    return clf.best_estimator_
