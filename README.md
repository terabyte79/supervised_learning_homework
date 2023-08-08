# supervised_learning_homework

# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

#### Explain the purpose of the analysis.
*  The purpose of this analysis is to determine the creditworthiness of borrowers.


#### Explain what financial information the data was on, and what you needed to predict.
* I was provided with a dataset containing historical lending activity data from a peer-to-peer lending services company. This dataset included information on both good loans (loans that were repaid) and bad loans (loans that were defaulted upon). My assignment was to utilize this historical lending data to develop a model capable of predicting the creditworthiness of potential future clients.


#### Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* "Credit risk presents a classification challenge that is inherently imbalanced. This is due to the fact that the number of healthy loans significantly outweighs the number of risky loans."

The total number of good loans information provided in the dataset: 75036.
The total number of bad loans information provided in the dataset: 2500.


#### Describe the stages of the machine learning process you went through as part of this analysis.
* The initial step involves reading the data into Python using the Pandas library. Subsequently, through Python functions, I separated the 'X' values representing input features from the 'y' values, which indicate the output label. Afterward, I utilized the 'train_test_split' function from the 'sklearn' Python library to partition my dataframe into training and testing sets.

I proceeded by invoking the logistic regression model, fitting it with the training data. Using this fitted model, I performed predictions for the 'y' values. Once the prediction phase concluded, I computed my model's effectiveness using pre-existing functions. This encompassed calculating the accuracy score, generating a confusion matrix, and producing a classification report to ascertain both precision and recall metrics for both good loans and bad loans.

Furthermore, an additional analysis was conducted utilizing resampled data employing the 'RandomOverSample' technique. This approach ensured that the quantity of good and bad loans provided to the model was equal.

#### Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

The Logistic Regression model was chosen as the machine learning approach for this assignment primarily due to its effectiveness in binary classification tasks. Logistic regression provides a probability score ranging from 0 to 1. It employs a distinctive curve known as the sigmoid curve. This curve initially starts at a low value close to 0, then rises rapidly, and eventually stabilizes as it approaches 1. During the training process, the model adjusts its parameters to fit the sigmoid curve.

Since the number of good loans outnumbers the bad loans, data resampling was performed using the RandomOverSampler from the imbalanced-learn library. Random Oversampler helps us achieve a more balanced representation of data by generating additional instances of the minority class (Bad loans) through duplication. This technique aims to ensure that both classes (Good loans and Bad loans) have a more equal presence in the dataset, thereby improving the performance of machine learning models, especially in cases where the minority class holds significant interest or importance.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

#### Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

#### Accuracy 
* Model 1 has an accuracy score of 99.18%

#### Good Loans -
* The tolal number of actual good loans = 18663 + 102 = 18765.
* Total number of loans predicted as good loans = 18663 + 56 = 18719.
* Total number of loans that were correctly predicted as good loans = 18663
* Precision = 18663/18719 = 0.997 = 1.00 ( Good loans) - The precision value is perfect.
* Recall = 18663 / 18765 = 0.994 = 0.99 - (Good loans ) - The recall values is perfect.
#### Bad Loans
* The tolal number of actual bad loans = 563 + 56 = 619
* Total number of loans predicted as bad loans = 102 + 563 = 665
* Total number of loans that were correctly predicted as bad loans = 563
* Precision = 563 / 665 = 0.846 = 0.85 (Bad loans) - Decent value
* Recall = 563 / 619 = 0.909 = 0.91 (Bad loans) - Decent value

The model performs exceptionally well in predicting good loans and reasonably well in predicting bad loans. Given that the amount of bad loan data used for training the model is significantly lower compared to good loans, the model, on the whole, achieves decent results, with a precision of 85% and a recall of 91%. Nevertheless, there is ample room for improvement.



#### Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

  #### Accuracy 
* Model 2 has an accuracy score of 99.38%

#### Good Loans -
* The tolal number of actual good loans = 18663 + 102 = 18765.
* Total number of loans predicted as good loans = 18649 + 4 = 18653
* Total number of loans that were correctly predicted as good loans = 18649
* Precision = 18649/18653 = 0.999 = 1.00 ( Good loans) - The precision value is perfect.
* Recall = 18649 / 18765 = 0.993 = 0.99 - (Good loans ) - The recall values is perfect.
#### Bad Loans
* The tolal number of actual bad loans = 4 + 615 = 619
* Total number of loans predicted as bad loans = 116 + 615 = 731
* Total number of loans that were correctly predicted as bad loans = 615
* Precision = 615/731 = 0.841 = 0.84 (Bad loans) - Decent value
* Recall = 615 / 619 = 0.993 = 0.99 (Bad loans) - Perfect value

The logistic regression model, fitted with oversampled data, excels at identifying all the bad loans. This is evident from the fact that out of 619 actual bad loans, it accurately predicted 615 as bad. However, it also misclassified 116 good loans as bad loans. Consequently, the recall value is nearly perfect at 99%, but the precision slightly decreases to 84%. The model will require further training to ensure that prospective good customers are not wrongly denied loans.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?

Model 2, utilizing oversampled data, demonstrates improved performance by accurately predicting the majority of bad loans. Among the 619 bad loans, Model 2 successfully predicted 615, signifying a substantial enhancement over the results obtained from Model 1. However, a limitation of Model 2 lies in its tendency to misclassify some good loans as bad. From a business standpoint, this could potentially result in missed opportunities, yet it also possesses the potential to safeguard the business from losses stemming from loan defaults.

A recommended approach involves segmenting the initial dataset based on loan amounts. Loans falling below a certain risk threshold that the institution is willing to accept could form one dataset, while loans exceeding this threshold could constitute another dataset. Alternatively, the business could maintain multiple datasets corresponding to different levels of loan amounts, factoring in various relevant considerations.

For the dataset encompassing loans within the business's comfort zone, a model could be trained where its  primary objective should be to accurately predict good loans. While the model should also identify bad loans, a degree of flexibility could be allowed in this prediction.

Conversely, when dealing with larger loan amounts, the model's emphasis should be on precisely predicting bad loans. This strategy enables the institution to mitigate the potential for significant losses effectively.



* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Yes, it's crucial to have a clear understanding of our objectives. If our main aim is to entirely mitigate the occurrence of bad loans, then achieving a substantial recall value for bad loans becomes paramount. Conversely, if our priority is to accurately predict good loans, our focus would be directed towards precision in our predictions.