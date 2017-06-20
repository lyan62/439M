clear;
clc;
dataLoading2;

%% test KNN without PCA
disp('testing KNN without PCA');
% use three neighbors for classification (has highest accuracy)
[class,accuracy] = KNNEval(3,tFeatures, tLabels, tstFeatures, tstLabels); 

%% test PCA_KNN
% use three neighbors, and first 100 components
disp('testing KNN with PCA');
[class,accuracy] = PCA_KNN(3,67,tFeatures,tstFeatures,tLabels, tstLabels); 

%% test LDA with KNN
disp('testing LDA followed by KNN');
[class,accuracy] = LDA_KNN(3,tFeatures,tstFeatures,tLabels, tstLabels);

%% test Bayes
disp('testing Bayes');
[class,accuracy] = bayesEval(tFeatures, tLabels, tstFeatures, tstLabels);

%% test PCA with Bayes
disp('testing PCA with Bayes');
[class,accuracy] = PCA_bayes(67,tFeatures,tstFeatures,tLabels, tstLabels);

%% test LDA with Bayes
disp('testing LDA with Bayes');
[class,accuracy] = LDA_bayes(tFeatures,tstFeatures,tLabels, tstLabels);

%% RESULT
%(train 10 test 3 for each class)
% testing KNN without PCA:
% accuracy =
%     0.5882

% testing KNN with PCA : 
% accuracy =
%     0.5833

% testing LDA followed by KNN 
% accuracy =
%     0.5784

% testing Bayes 
% accuracy =
%     0.6814

% test PCA with Bayes
% accuracy =
%     0.7255

% test LDA with Bayes 
% accuracy =
%     0.7794


%%%%%%(train 8 test 5 for each class)
% testing KNN without PCA
% accuracy =
%     0.7059
% 
% testing KNN with PCA 
% accuracy =
% 
%     0.7206
% 
% testing LDA followed by KNN
% accuracy =
% 
%     0.7294
% 
% testing Bayes 
% accuracy =
% 
%     0.8029
% 
% testing PCA with Bayes
% accuracy =
% 
%     0.8412
% 
% testing LDA with Bayes
% accuracy =
% 
%     0.8529
