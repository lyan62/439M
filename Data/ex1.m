%% divide data into trainning and testing set
% use LBPFeatures
load 'Data.mat'


for i  = 1: 200
    % trainning set: 400 faces
    trainning_faces{2*i-1} = face(:,:,3*i-2);
    trainning_faces{2*i} = face(:,:,3*i-1);
    % testing set: 200 faces
    testing_faces{i} = face(:,:,3*i);
    testing_features(i,:) = extractLBPFeatures(testing_faces{i});  
    testing_labels(i,1) = i;
end

% obtain 1*59 feature vector for each image
for l  = 1: length(trainning_faces)
    training_features(l,:) = extractLBPFeatures(trainning_faces{l});
    trainning_labels(l,1) = ceil(l/2);
end



%% 
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

% obtain 1*59 feature vector for each image
for l  = 1: length(trainning_faces)
    tFeatures(l,:) = reshape(trainning_faces{l},1,s); 
    tLabels(l,1) = ceil(l/2);
end
