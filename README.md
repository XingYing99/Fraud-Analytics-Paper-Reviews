# Fraud-Analytics

# Paper 1 Fraud Detection using Machine Learning and Deep Learning
https://ieeexplore.ieee.org/document/9004231

#### Summary
This paper seeks to understand and evaluate the different Machine Learning models in detecting fraudulent behavior in credit cards. Different Machine Learning and Deep Learning models are thus being benchmarked and compared, to assess which type of model returns the most accurate results according to the size of the dataset.
 
#### Problem Statement
Currently, it is difficult to train models while ensuring its efficiency due to noisy data and overlapping patterns. Furthermore, the dynamics and fraudulent behavior is ever-changing due to emerging new ways to commit fraud and misuse data, and thus models have to also consistently learn and pick-up these new patterns and behaviors.
 
Furthermore, the lack of real-world fraud data due to privacy and confidentiality issues forces researchers and engineers to solely rely on the use of other datasets or datasets which try to imitate fraud data. These datasets may sometimes be highly inaccurate due to unbalanced datasets, and the computation time could be very long.
 
As such, models are not able to effectively predict fraud data. However, as fraudulent transactions are a very prevalent problem especially in our technologically-advanced world today, we aim to find the best model or mix of models to do so.
 
#### Main Results
The paper conducted experiments on datasets of different sizes, using models such as Support Vector Machine (SVM), Convolutional Neural Network (CNN), Random Forest and K-Nearest Neighbors algorithm. It also combined and took ensemble approaches of the different models to test if there would be any differences in the results. The paper also tested AutoEncoders and Restricted Botlzmann machines (RBM) to evaluate its use dependent on dataset and in an unsupervised learning setting. 
 
In general, SVM was the consistently best performing model, while CNN was the best deep learning method. KNN provided good results with both large and small datasets, while Random Forest worked best with small datasets. For the large datasets especially, a combination of SVM and CNN would provide the most reliable performance. Deep CNN was also found to train almost twice faster than the rest of the models.
 
However, in general, these models were only tested in a supervised learning context and may not be applicable in real life. Autoencoders could also be used as it is useful for the labelling of datasets, however it is costly.
 
#### Analysis of the results
To better understand the results, we ran the models on a large dataset to test for its accuracy.

Since the paper used a grid search approach to find the best parameter for the calculation of the results, we decided to run our own model by switching the parameters based on our understanding of  the data by looking at the correlation matrix and the scatter plot to determine the number of fraudulent data points. After rebalancing the classes and training the model, we found that using SVM, the accuracy of detecting frauds is 0.993 while the probability of detecting fraud is 0.874 as we detected 174/199 total frauds. 
 
Using a separate CNN model, we found that the accuracy was low at 0.923.

We decided to explore other methods to test if other models would be more accurate or have the same accuracy and precision. As such, we ran an Artificial Neural Network (ANN) model on the dataset, and we found that it had a precision value of 0.873 and an accuracy value of 94.9%  

However, some limitations we found from the paper was that the datasets were not explained enough. We were not given a clear idea of the type of data and the usage of each datapoint, as well as the differences in the three datasets mentioned in the paper. It was thus difficult to reach a conclusive standpoint regarding the usage of the different Machine Learning models in relation to the data. 

Furthermore, there were no assumptions mentioned in the paper, and we are not certain if there are certain data points excluded or added in the calculation and derivation of the results. As a result, the findings shown in the paper may not have been the most accurate. Nonetheless, after running our own tests and models, we have found that our results align to that of the paper. 

# Paper 2 An experimental study with imbalance classification approaches for credit card fraud detection
https://ieeexplore.ieee.org/document/8756130

#### Problem Statement
Identifying weaknesses of existing class imbalance approaches; more specifically, the machine learning methods, so that we can develop a highly efficient solution for the class imbalance problem

####  Main Results from the paper
The dataset used contains 8 variables namely custID, gender , state  and cardholder, balance, numTrans, numIntTrans ,creditLine. 

#### Assumptions
Due to limitations in computational power, the paper only uses 2% of the data but maintains the same imbalance ratio at the same time. Performance metrics  used for comparison include Accuracy, Sensitivity, and area under the precision-recall curve (AUPRC). Experiment results in this paper are  obtained using R programming. In addition, multicollinearity exists between variables as there are 2 variables with high variance inflation factor, hence , some algorithms are not used in stage 2 of the algorithms

The paper reflects the experimentations and the performance in using 8 different machine learning models for fraud detection. By comparing the performance in Fig. A , only 3 models are chosen for the next stage (C5.0(Decision tree), SVM, ANN )  as the other algorithms either perform worse than these 3 models or the models are not applicable since the data violates some assumptions related to the models. Afterwards, various class imbalance approaches such as Random Oversampling (RO) and Cost-Sensitive models are applied  to these algorithms with the aim to improve performance by reducing the class imbalance issue.

#### Analysis of results in the paper
Among the 3 metrics , only the sensitivity value increases from 39% to 65% for cost sensitive SVM, 43% to 66% random oversampling C5 and 43% to 66% cost sensitive C5. The AUPRC either decreases or maintains the same value with the base-line model while all the accuracy decreases after applying the class imbalance approach. Therefore the paper states that some of the class imbalance approaches normally used to solve imbalance problems may be ineffective when the imbalance is extreme, such as generating a significant number of false positives which causes the AUPRC and accuracy to decrease. Also, the paper showed that considering just one performance measure for imbalanced learning is misleading. For instance , high accuracy in this situtuation does not indicate good performance as the low recall indicates that most of the fraud cases are not (identified) predicted. 

#### Implementation and experiments

a) Implement the framework inside the paper (Class ratio 15:1)
Assumptions
In this experiment we have maintained the same class imbalance ratio and data size. The algorithms we are using include SVM, Decision Tree, ANN(Neural Network), XGBoost and Random Forest while the imbalance classification approaches used will be random oversampling and cost sensitive models since these are the 2 approaches that will increase the sensitivity value in the paper. The results might be a bit different from the paper as the parameters used in the paper are unknown.
Analysis of results
Based on the results in table 1, only cost sensitive SVM leads to an increase in both the AUPRC and sensitivity while maintaining a similar accuracy. Other imbalance classification approaches did increase the sensitivity value but the AUPRC is still similar to the AUPRC in the base-line models. Hence, we feel that the increase in AUPRC is likely due to the decrease in precision. In this situation, the performance of the model did not increase overall as low prediction indicates that very few of the fraud cases detected are true.

b) Experiments to verify if the statement “ Class imbalance approaches normally used to solve imbalance problems may be ineffective when the imbalance is extreme” is true. (Class ratio 10:1)

####  Assumptions
The class ratio is changed to 10:1 by increasing the number of fraud cases from the original data using random selection while maintaining the same sample data size. The algorithms and the imbalance classification approaches including the sampling strategy and class weights used will be the same as the previous experiment. 
Analysis of results
Based on the results in table 2, only decision-tree with over sampling results in an increase in both AUPRC and recall while maintaining the same accuracy, hence leading to an increase in overall performance. Other approaches with SVM or decision tree either result in an increase in recall while maintaining the same auprc, or result in a decrease in recall but an increase in auprc. Except for oversampling , other class imbalance approaches did not improve the overall performance of the models for this dataset. Hence, we conclude that the reason for the inefficiency of the imbalance classification approaches is not due to the extreme imbalance ratio since a smaller imbalance ratio did not increase the overall performance of the models.

c) Experiments apply similar algorithms and imbalance classification approaches on a different dataset.
Dataset
The dataset used here consists of numerical values from 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28 and 2 numerical features “Time” and “Amount” that are not transformed. In the experiment, we will scale the “Time “ and “Amount” just like other variables. The class imbalance ratio for this dataset is 577:1.




#### Assumptions:
Based on the heatmap and the multicollinearity test, we assume that there exist no multicollinearity among variables.In this experiment , we will use similar methods in the paper to ensure that the results are comparable.
Analysis of results
Based on the results shown in table 3, cost-sensitive decision-tree and  random oversampling have increased both the recall and AUPRC while maintaining the same accuracy value.

#### Conclusion
In conclusion, the statement “Class imbalance approaches normally used to solve imbalance problems may be ineffective when the imbalance is extreme” stated in the paper may not always be true as the strategy of oversampling with decision tree and cost-sensitive decision tree did increase the overall performance for the other credit card dataset that has a imbalance ratio of 577:1. In addition, only the oversampling strategy leads to an increase in overall performance when the  imbalance ratio in the paper's dataset  is reduced. Hence, we conclude that the effectiveness of the imbalance classification approaches is dependent on the nature and the variables in the dataset and not the imbalance ratio.

#### Limitation
Due to limited computational power and time, the parameters used for grid search in the algorithms are very limited as some models such as SVM take very long to run. In addition, since we are only sampling 2% of the data from the raw dataset , a large amount of information will be lost for model training . Therefore , we are unable to obtain the best results from our experiments on the dataset provided by the paper. Also, the imbalance classification approaches used in the experiments are limited, hence the conclusion in the experiment can only be applied to the approaches used and not all imbalance classification approaches.

#### Future work
We can try other approaches such as random under-sampling and SMOTE using KNN. On top of that, we should also include F1 score as one of the evaluation metric as we will want to detect most of the fraud cases(recall)  whilst keeping the cost at which this is achieved under control (precision) as false positive cases will also lead to human labour costs in verifying the client’s data.

# Paper 3:  A Multiple Classifiers System for Anomaly Detection in Credit Card Data with Unbalanced and Overlapped Classes https://ieeexplore.ieee.org/document/8985298

#### Problem Statement 
Credit card fraud and default payments are two key anomalies in credit card transactions, and one of the ways to tackle the issue is data mining approaches. However, the credit card data might have unbalanced class distribution or overlapping of class samples, which both can lead to low detection rates for the anomalies. Also, general learning algorithms are biased to the majority class samples, which makes it harder to detect the anomalies, which are likely the minorities.

#### Main Results
In the paper, there were five basic learning algorithms discussed, followed by the Multiple Classifiers System (MCS), which is the key solution explored. The MCS is a combination of a set of classifiers or learning algorithms, and there are three combination strategies when employing MCS:
Sequential Combination
Parallel Combination
Hybrid Combination

Sequential Combination involves using two or more single classifiers and processing the input data sequentially. Typically, simple classifiers will be utilized first, followed by the more complex ones. Parallel Combination involves processing the dataset with singular classifiers, and then combining the output from all the classifiers to get the eventual outcome and result. An example is Bagging with Random Forest as the base classifier. Lastly, Hybrid Combination puts both the Sequential Combination and Parallel Combination together. The first classifier would take in the input data and parse the output into several parallel classifiers. A single combination function will then merge the output of the individual parallel classifiers.

The main evaluation measure used in the paper is the Sensitivity or True Positive Rate (TPR), to identify the classification rate for both majority and minority classes, in which the minority classes of frauds and credit defaults are being focused on.

Two datasets were used in the paper, namely the Credit Card Fraud (CCF) data set and the Credit Card Default Payment (CCDP) data set. As the CCF dataset provides a more unbalanced dataset with samples of both classes overlapping, our group will focus more on the exploration of this dataset.
The CCF has a total of 284,807 transactions and is highly unbalanced with 492 fraud transactions which makes up 0.173% of the dataset. Due to the confidentiality of credit information, the majority of the fields are encoded as V1,V2, up to V28. 

The algorithm implemented in the paper is the sequential combination which includes 2 classifiers. The non-fraud cases predicted by the first classifier (C4.5 , known as decision tree) will be taken out and passed to the second classifier (Naive Bayes) as the test data for classification again. As a result, we will combine the predicted fraud cases from the 2 classifiers and the predicted non-fraud cases from the second classifier as our final output. From the results shown in the paper , we can see that the TPR for CCF has increased from 0.829 to 0.872 when both C4.5 and naive bayes are used sequentially instead of just the Naive Bayes classifier. A key assumption here for the Naive Bayes classifier is that the variables in the dataset are independent of each other.

#### Analysis of Results

In the paper, the researchers hypothesized that single classifiers were weak against classifying data sets that contain unbalanced class distribution and overlapping classes. They managed to prove it with their experimentations, where they displayed the TPR values of the single classifiers which were lower than the sequentially combined algorithm. The sequentially combined algorithm used was the C4.5 and NB as the first and second classifier respectively. The C4.5 algorithm performed the best in producing the highest TPRs for the majority class of both datasets, while the NB algorithm performed the best in producing the highest TPRs for the minority class of both datasets.

The increase in TPR is due to an increase in True Positives (TP) and decrease in False Negatives (FN). However, the number of False positives increased from 26 to 1834 and the True positive increased from 112 to 126 when we compare the results obtained by a single Decision Tree classifier, as seen in Fig. C and Fig. D, leading to a lower precision value which may be a concern as it indicates a high number of false positive predicitons. Therefore, we infer that the increase in 14 true positive cases predicted came with the cost of additional 1808 false positive cases predicted.

The group felt that the work presented in the paper was quite comprehensive and showed strong credibility as our own implementation achieved similar results. In the paper, the researchers showcased the proposed MCS alongside other researchers’ work on both datasets, and showed how the results were better in terms of the TPR or Sensitivity. On top of TPR, our group would strongly recommend that precision should be heavily considered as a key metric value, as it would be able to focus on the small positive class, considering that the main focus of the paper is to find the correctly detected positive samples.

As mentioned in the paper, Sequential Combination was the main multiple classifier strategy that was adopted. In the future, they are looking at other combination strategies, mainly the hybrid combination. Using deep learning algorithms such as the Long Short-Term Memory (LSTM) would provide more layers and recurring learning opportunities for the model to further improve itself based on past results.

### Summary and Analysis of the Three Papers

In all 3 papers, the main problem that the papers were trying to solve were the credit card fraud classification problem, with paper 2 and 3 going more in depth to solve further deeper issues such as unbalanced datasets (paper 2 and paper 3) and overlapping samples (paper 3).

In paper 1, the different models and approaches for different sizes of datasets were explored and the results varied based on the size of the datasets. SVM and CNN tend to perform better for larger datasets, while models like Random Forest tend to perform better for smaller datasets. This is further supported by the results in paper 2 and 3 where the datasets used were considered large, and the models that performed better and worse were the SVM models and Random Forest respectively.

Both the imbalance classification approaches in paper 2 and sequential combination strategy in paper 3 results in an increase in the recall value or TPR, as shown in the results which is the most important problem that we should address as the inability to detect fraud cases will incur huge financial losses for the company.  However, the high number of false positive cases predicted by the sequential combination in paper 3 will result in the company incurring high labour cost in eliminating all the false positive cases predicted. If the cost of labour were to exceed the cost of not detecting additional fraud cases, the approach in paper 3 will not be cost-efficient. Hence, we propose 2 approaches in resolving these challenges.

One approach that can be applied in paper 2 and 3 collectively is the precision-recall balance approach, which will help optimize a cost function, consisting of the cost of fraud and the cost of monitoring. Afterwards, we can look at the computed class probabilities by each classifier and shift the threshold at which we deem a case to be fraud. This allows us to further tune the cost-efficiency of our models. This approach was attempted by Robin Teuwens in search of minimizing the cost function, and also proving the importance of the selection of a performance metric when it comes to datasets with unbalanced classes, which is highly applicable to our datasets (Teuwens, 2019). 

Another approach will be the use of paper 2’s approach of oversampling on paper 3’s datasets, to further solve the unbalanced class issue for the CCF data since the oversampling strategy has resulted in an overall increase in both the recall and precision while maintaining the same accuracy.  This can be done using SMOTE or other oversampling strategies before using the sequential combination approach to classify the data. Other algorithms such as Long Short Term Memory (LSTM) can also be implemented in this approach so that we can achieve better results.

#### Results
![image](https://user-images.githubusercontent.com/71431944/157276365-d008a74e-15a6-40d3-9a38-b62c6e589ab4.png)

![image](https://user-images.githubusercontent.com/71431944/157276480-cc0dce27-f802-4200-a49e-df35523c849a.png)

![image](https://user-images.githubusercontent.com/71431944/157276537-a220317a-4ab0-463b-a5f8-38e2be81bc21.png)
