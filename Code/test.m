%% test
%load data;
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
[class,accuracy] = KNNEval(1,tFeatures, tLabels, tstFeatures, tstLabels);

%% test KNN with PCA
%[projected_t,PC,V] = pca1(tFeatures);
%projected_tst = pca1(tstFeatures);

features = [tFeatures; tstFeatures];
projected = pca1(features');
projected_t = projected(:,1:400)';
projected_tst = projected(:,401:end)';

[class,accuracy] = KNNEval(1,projected_t, tLabels, projected_tst, tstLabels);


%%
features = [tFeatures; tstFeatures];
[coeff,projected]= pca(features);
projected_t = projected(1:400,1:200);
projected_tst = projected(401:end,1:200);
[class,accuracy] = KNNEval(1,projected_t, tLabels, projected_tst, tstLabels);

%% test my PCA
features = [tFeatures; tstFeatures]; %[600 * 504]
features = features'; % need [m*n] for applyPCA
projected = applyPCA(features); %  Y = [504*600]
projected = projected'; %[600 * 504]
% reduce dimension to 200(use first 200 principal components)
projected_t = projected(1:400,1:200);
projected_tst = projected(401:end,1:200);
[class,accuracy] = KNNEval(1,projected_t, tLabels, projected_tst, tstLabels);

%% test PCA_KNN
% use three neighbors, and first 100 components
[class,accuracy] = PCA_KNN(3,100,tFeatures,tstFeatures,tLabels, tstLabels); 





