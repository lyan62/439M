%% test_ver2
%% data processing
clear
clc
load 'Data.mat'
s = size(face,1)*size(face,2);
for i  = 1: 200
    % trainning set: 400 faces
    trainning_faces{2*i-1} = face(:,:,3*i-2);
    trainning_faces{2*i} = face(:,:,3*i-1);
    % testing set: 200 faces
    testing_faces{i} = face(:,:,3*i);
    tstFeatures(i,:) = reshape(testing_faces{i},1,s);  
    tstLabels(i,1) = i;
end

%  reshap image into feature vectors for each image
for l  = 1: length(trainning_faces)
    tFeatures(l,:) = reshape(trainning_faces{l},1,s); 
    tLabels(l,1) = ceil(l/2);
end

%% test KNN without PCA
% use three neighbors for classification (has highest accuracy)--obtained accuracy = 0.6150
[class,accuracy] = KNNEval(3,tFeatures, tLabels, tstFeatures, tstLabels); 
%accuracy = 0.6150
%% test PCA_KNN
% use three neighbors, and first 100 components--obtained accuracy = 0.6000
[class,accuracy] = PCA_KNN(3,100,tFeatures,tstFeatures,tLabels, tstLabels); 
% accuracy = 0.6000
%% test LDA with KNN
[class,accuracy] = LDA_KNN(3,tFeatures,tstFeatures,tLabels, tstLabels);
% accuracy = 0.6200

%% test Bayes
disp('testing Bayes');
[class,accuracy] = bayesEval(tFeatures, tLabels, tstFeatures, tstLabels);
% accuracy = 0.6400

%% test PCA with Bayes
[class,accuracy] = PCA_bayes(100,tFeatures,tstFeatures,tLabels, tstLabels);
% accuracy = 0.6350
%% test LDA with Bayes
[class,accuracy] = LDA_bayes(tFeatures,tstFeatures,tLabels, tstLabels);
% accuracy = 0.6400
