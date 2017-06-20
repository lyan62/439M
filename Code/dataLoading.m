%% data processing
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