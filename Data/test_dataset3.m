clear;
clc;
dataLoading3;

%% test KNN without PCA
disp('testing KNN without PCA');
% use three neighbors for classification (has highest accuracy)
[class,accuracy] = KNNEval(1,tFeatures, tLabels, tstFeatures, tstLabels); 

%% test PCA_KNN
% use three neighbors, and first 100 components
disp('testing KNN with PCA');
[class,accuracy] = PCA_KNN(1,67,tFeatures,tstFeatures,tLabels, tstLabels); 

%% test LDA with KNN
disp('testing LDA followed by KNN');
[class,accuracy] = LDA_KNN(1,tFeatures,tstFeatures,tLabels, tstLabels);

%% test Bayes
disp('testing Bayes');
[class,accuracy] = bayesEval(tFeatures, tLabels, tstFeatures, tstLabels);

%% test PCA with Bayes
disp('test PCA with Bayes');
[class,accuracy] = PCA_bayes(100,tFeatures,tstFeatures,tLabels, tstLabels);

%% test LDA with Bayes
disp('test LDA with Bayes');
[class,accuracy] = LDA_bayes(tFeatures,tstFeatures,tLabels, tstLabels);


%% Result:
% testing KNN without PCA
% accuracy = 
%     0.9538
% 
% testing KNN with PCA
% accuracy =
%     0.9013
% 
% testing LDA followed by KNN
% accuracy =
%     0.8214
%
% testing Bayes
% accuracy =
%      1
% 
% test PCA with Bayes
% accuracy =
% 
%      1
% 
% test LDA with Bayes
% accuracy =
% 
%      1