# Credit Risk Analysis

This data analysis applies supervised machine learning models to analyze credit card risk. Regardless of the problem type, in machine learning, we follow a familiar 
paradigm: Model → Fit (Train) → Predict

Because credit risk is an unbalanced classification problem (good loans typically outnumber risky loans), there is a need to apply machine learning techniques to train and evaluate models that work with unbalanced classes.  

## Overview

This data analysis builds and evaluates six machine learning models. It leverages both the imbalance-learn and skit-learn machine learning libraries.

Pre-work was also performed to create the Python machine learning environment and to pre-process the data for machine learning: clean the data, encode the data using pandas get_dummies(), and scale the data.

### Objective 1 - Use Resampling Models to Predict Credit Risk

First RandomOverSampler and SMOTE algorithms are used and then lastly the Undersampling ClusterCentroids alogrithm is used for this objective. 

Each of these algorithms train a logistic regression classifier. Logistic regression classification algorithms are used to predict a discrete set of classes or categories (for example, Yes/No, Young/Old, Happy/Sad).

Logistic regression can be used to predict which category or class a new data point should belong to. In this work, it is being used to predict credit risk: high or low.

Using these three algorithms the following steps are performed:

1. Resample the dataset
2. View the count of target classes
3. Train a logistic regression classifier
4. Calculate the balanced accuracy score
5. Generate a confusion matrix
6. Generate a classification report.

#### Resampling Images

##### Naive Random Oversampling counter results:

![NRO counter image](/resources/NRO_counter_image.png)

##### SMOTE Oversampling counter results:

![SMOTE counter image](/resources/SMOTE_counter_image.png)

##### Undersampling counter results:

![Undersampling counter image](/resources/Undersampling_counter_image.png)


### Objective 2 - Use SMOTEENN algorithm to Predict Credit Risk

For this objective, a combinational approach of both over and under sampling with the SMOTEENN algorithm is used with the intention of determining if the results from this combinational approach are better at predicting credit risk than the resampling algorithms used with Objective 1.

Using the SMOTEENN algorithm the following steps are again performed:

1. Resample the dataset
2. View the count of target classes
3. Train a logistic regression classifier
4. Calculate the balanced accuracy score
5. Generate a confusion matrix
6. Generate a classification report.

Note: Counters weren't calculated as part of this objective.

### Objective 3 - Use Ensemble Classifiers to Predict Credit Risk

For the third objective, two different ensemble classifiers are trained to predict credit risk, tested and then evaluated. 

Using both of algorithms the following steps are performed (note the difference in objective 3 for step 3):
1. Resample the dataset
2. View the count of target classes
3. Train the ensemble classifier
4. Calculate the accuracy score
5. Generate a confusion matrix
6. Generate a classification report.

Note: Counters weren't calculated as part of this objective.

### Objective 4 - Summarize the analysis of the performance of all the machine learning models used 

This README.md is intended to fulfill Objective 4 and provide an overview, results, and summary of this work.


#### Project Resources: 

* Python with pandas, numpy, pathlib, collections, sklearn and imblearn 

## Results:

As part of this work, resampling is done in some cases and then results are generated for each of the machine learning algorithms. 

These results are provided through a balanced accuracy score, a confusion matrix, and a classification report. 

Testing and training scores are also generated to help understand whether the models are overfit or underfit.

### Balanced accuracy score

The balanced accuracy score computes the balanced accuracy, which avoids inflated performance estimates on imbalanced datasets. It is the macro-average of recall scores per class or, equivalently, raw accuracy where each sample is weighted according to the inverse prevalence of its true class. Thus for balanced datasets, the score is equal to accuracy. The balanced accuracy is calculated in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

The best value is 1 and the worst value is 0 when adjusted=False. Default adjusted=False.

#### Naive Random Oversampling balanced accuracy score:

![NRO balanced accuracy score image](/resources/NRO_balanced_accuracy_score_image.png)

#### SMOTE balanced accuracy score:

![SMOTE accuracy score image](/resources/SMOTE_balanced_accuracy_score_image.png)

#### Undersampling balanced accuracy score:

![Undersampling balanced accuracy score image](/resources/Undersampling_balanced_accuracy_score_image.png)

#### SMOTEENN balanced accuracy score:

![SMOTEEN balanced accuracy score image](/resources/SMOTEENN_balanced_accuracy_score_image.png)

#### Balanced Random Forest Classifier balanced accuracy score:

![Balanced Random Forest Classifier balanced accuracy score image](/resources/balanced_random_forest_balanced_accuracy_score_image.png)

#### Easy Ensemble AdaBoost Classifier balanced accuracy score:

![Easy Ensemble AdaBoost Classifier balanced accuracy score image](/resources/easy_ensemble_adaboost_balanced_accuracy_score_image.png)

### Confusion matrix

A confusion matrix compares the predicted values from a model against the actual values. The entries in the confusion matrix are the number of true positives (TP), true 
negatives (TN), false positives (FP), and false negatives (FN). 

The confusion matrix allows the use of 4 important measure formulas on our models: accuracy, precision, sensitivity, and F1 scores.

#### Naive Random Oversampling logistic regression confusion matrix:

![NRO confusion matrix image](/resources/NRO_confusion_matrix_image.png)

#### SMOTE logistic regression confusion matrix:

![SMOTE confusion matrix image](/resources/SMOTE_confusion_matrix_image.png)

#### Undersampling logistic regression confusion matrix:

![Undersampling confusion matrix image](/resources/Undersampling_confusion_matrix_image.png)

#### SMOTEENN logistic regression confusion matrix:

![SMOTEENN classifiction report image](/resources/SMOTEENN_confusion_matrix_image.png)


#### Balanced Random Forest Classifier confusion matrix:

![Balanced Random Forest Classifier confusion matrix image](/resources/balanced_random_forest_confusion_matrix_image.png)

#### Easy Ensemble AdaBoost Classifier confusion matrix:

![Easy Ensemble AdaBoost Classifier confusion matrix image](/resources/easy_ensemble_adaboost_confusion_matrix_image.png)


#### Model accuracy
Accuracy is the percentage of correct predictions. Accuracy = TP + TN / (TP + TN + FP + FN). 

#### Model precision
Precision is a measure of how reliable a positive classification is. Stated another way, precision is the percentage of positive predictions that are correct. Precision = TP / (TP + FP).

#### Model sensitivity/recall
Sensitivity (or Recall) is the percentage of actual positive results that are predicted correctly. Sensitivity = TP / (TP + FN). It is sometimes called the True Positive Rate. Sensitivity is a measure of how many observations with a positive condition will be correctly classified.

#### Model F1 score

The F1 score (also called the harmonic mean) balances precision and sensitivity. F1 = 2(Precision * Sensitivity) / (Precision + Sensitivity)

### Classification report

The classification report is used to calculate these measures but it doesn't know which class is positive and which is negative, this it gives the scores for both classes—"0" and "1"

#### Naive Random Oversampling logistic regression classification report:

![NRO balanced classification report image](/resources/NRO_balanced_classification_report_image.png)

#### SMOTE logistic regression classification report:

![SMOTE classification report image](/resources/SMOTE_classification_report_image.png)

#### Undersampling logistic regression classification report:

![Undersampling classifiction report image](/resources/Undersampling_classification_report_image.png)

#### SMOTEENN logistic regression classification report:

![SMOTEENN classifiction report image](/resources/SMOTEENN_classification_report_image.png)

#### Balanced Random Forest Classifier classification report:

![Balanced Random Forest Classifier classification report image](/resources/balanced_random_forest_imbalanced_classification_report_image.png)

#### Easy Ensemble AdaBoost Classifier classification report:

![Easy Ensemble AdaBoost Classifier classification report image](/resources/easy_ensemble_adaboost_classification_report_image.png)

### Feature ranking by feature importance

As part of the Balanced Random Forest Classier work feature ranking by feature importance was performed and a snapshot is shown here:

![BRFC feature rank image](/resources/balanced_random_forest_features_sorted_desc_by feature_importance_image.png)

### Training and testing scores

If the training score and the testing(validation) score are both low, the estimator will be underfitting. If the training score is high and the testing (validation) score is low, the estimator is overfitting and otherwise it is working very well. A low training score and a high validation score is usually not possible. 

#### Naive Random Oversampling logistic regression training and testing scores:

![NRO training testing score image](/resources/NRO_training_testing_scores_image.png)

#### SMOTE logistic regression training and testing scores:

![SMOTE training testing score image](/resources/SMOTE_training_testing_scores_image.png)

#### Undersampling logistic regression training and testing scores:

![Undersampling training testing score image](/resources/Undersampling_training_testing_scores_image.png)

#### SMOTEENN logistic regression training and testing scores:

![SMOTEENN training and testing scores image](/resources/SMOTEENN_training_testing_scores_image.png)

#### Balanced Random Forest Classifier training and testing scores:

![Balanced Random Forest Classifier training and testing scores image](/resources/balanced_random_forest_training_testing_scores_image.png)

#### Easy Ensemble AdaBoost Classifier training and testing scores:

![Easy Ensemble AdaBoost Classifier training and testing scores image](/resources/easy_ensemble_adaboost_training_testing_scores_image.png)

#### Underfitting and Overfitting
* Underfitting is a scenario in data science where a data model is unable to capture the relationship between the input and output variables accurately, generating a high error rate on both the training set and unseen data. A model this is underfitted means it is too simple with too few features and too little data to build an effective model. , An underfit model has high bias and low variance.
* Overfitting is a modeling error in statistics that occurs when a function is too closely aligned to a limited set of data points. As a result, the model is useful in reference only to its initial data set, and not to any other data sets. A model that is overfitted may be too complicated, making it ineffective. An overfit model has low bias and high variance.

### Key Results Call Outs:

To help understand these results, this list describes the balanced accuracy score and the precision and recall scores for each of the six machine learning models:
* Using Naive Random Oversampling with a logistic regression model resulted in a balanced accuracy score of 64%, a precision score of 1% , and a recall score of 62%. This model has an F1 score of 2%. This model has a false positive ratio of 56%.
* Using SMOTE Oversampling with a logistic regression model resulted in a balanced accuracy score of 63%, a precision score of 1%, and a recall score of 62%. We also have an F1 score of 2%. This model has a false positive ratio of 36%.
* Using Undersampling using ClusterCentroids with a logistic regression model resulted in a balanced accuracy score of 51%, a precision score of 1%, and a recall score of 59%.  We also have an F1 score of 1%. This model has a false positive ratio of 57%.
* Using Combination (Over and Under) Sampling using SMOTEENN and a logistic regression model resulted in a balanced accuracy score of 64%, a precision score of 1%, and a recall score of 70%. This model has a false positive ratio of 42%.
* Using the Balanced Random Forest Classifier model resulted in a balanced accuracy score of 79%, a precision score of 3%, and a recall score of 70%. This model has a false positive ratio of 13%.
* Using the Easy Ensemble AdaBoost Classifier model resulted in a balanced accuracy score of 92%, a precision score of 6%, and a recall score of 89%. This model has a false positive ratio of 10%.

## Summary:

Selecting a model revolves around more than either accuracy or precision. No model is going to be 100% perfect and no matter which model selected there will be a borrower that will default.

### Points considered:
* Balanced accuracy scores range from 51% to 92%. 
* High precision and high recall mean we have a good and balanced model. None of our models are good and balanced.
* High precision and low recall means we have a model that is not good in detection but accurate when it does. None our models have high precison scores. Precision scores range for 1% to 6%. None of the models are very precise.
* Low precision and high recall means we have a model that has good detection capability but chances of false positives are high Recall scores range from 59% to 89%. Considering we are talking about credit risk, we need to look at further measures in consideration of these models.

One idea I had, was that we could also determine what percentage of false positives (what false positive ratio) would be tolerable. False positive ratio = FP/FP+TN. Let's say we determine a false positive ratio of 25% is tolerable and
then let's look at a model's helpfulness from that perspective in addition to it's other useful measures. Both the Balanced Random Forest Classifier model and the Easy Ensemble AdaBoost Clasifier model have false positive ratios below 25% so using those model could help us stay well below this target 25%. We may give up more loan candidates than our target though because these models are providing false positive ratios below the 25% range. (13%, 10%) 

I would not recommend using Naive Random Oversampling with a logistic regression model mostly because it's higher false positive ratio.
Instead of picking one model, I think I would start out using 3 of the models in combination: SMOTE Oversampling with a logistic regression model, the Balanced Random Forest Classifier model, and the Easy Ensemble AdaBoost Clasifier model. I would gather more data (hoping to get more high risk samples to train with) and see if that would improve any of the models further by training and testing on a larger set of data. 

### Training and testing scores check in
Looking at the training and testing scores, it is hard to say whether doing more training and testing would improve the models. None of the models are showing as overfit or underfit at this time. The Undersampling using ClusterCentroids with a logistic regression model did have a 44% testing score which isn't a good testing score compared to its 67% training score.