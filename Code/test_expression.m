clear
clc
load 'Data.mat'
s = size(face,1)*size(face,2);
for i  = 1: 200
    % trainning set: 400 faces
    % trainning expression and illumination
    trainning_faces{2*i-1} = face(:,:,3*i-2);
    trainning_faces{2*i} = face(:,:,3*i);
    % testing set: 200 faces
    testing_faces{i} = face(:,:,3*i-1);
    tstFeatures(i,:) = reshape(testing_faces{i},1,s);  
    tstLabels(i,1) = i;
end

%  reshap image into feature vectors for each image
for l  = 1: length(trainning_faces)
    tFeatures(l,:) = reshape(trainning_faces{l},1,s); 
    tLabels(l,1) = ceil(l/2);
end

%% test KNN without PCA
disp('testing KNN without PCA');
% use three neighbors for classification (has highest accuracy)
[class,accuracy] = KNNEval(3,tFeatures, tLabels, tstFeatures, tstLabels); 

%% test PCA_KNN
% use three neighbors, and first 100 components
disp('testing KNN with PCA');
[class,accuracy] = PCA_KNN(3,100,tFeatures,tstFeatures,tLabels, tstLabels); 

%% test LDA with KNN
disp('testing LDA followed by KNN');
[class,accuracy] = LDA_KNN(3,tFeatures,tstFeatures,tLabels, tstLabels);


%% test Bayes
disp('testing Bayes');
[class,accuracy] = bayesEval(tFeatures, tLabels, tstFeatures, tstLabels);

%% test PCA with Bayes
disp('test PCA with Bayes');
[class,accuracy] = PCA_bayes(100,tFeatures,tstFeatures,tLabels, tstLabels);

%% test LDA with Bayes
disp('test LDA with Bayes');
[class,accuracy] = LDA_bayes(tFeatures,tstFeatures,tLabels, tstLabels);

%%

% testing KNN without PCA
% accuracy =
% 
%     0.6500
% 
% testing KNN with PCA
% accuracy =
% 
%     0.6450
% 
% testing LDA followed by KNN
% accuracy =
% 
%     0.6600
%
% testing Bayes
% accuracy =
%     0.6650
%
% test PCA with Bayes 
% accuracy =
%     0.6700
%
% test LDA with Bayes
% accuracy =
%     0.6650