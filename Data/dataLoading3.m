clear
clc
%% preprocessing dataset3
% train with first 14 illuminations in each class
% test with rest 7
load 'illumination.mat';
num_class = size(illum,3);
for i  = 1: num_class
    for n  = 1:14
        ind = (i-1)*14 + n;
        trainning_faces(:,ind) = illum(:,n,i);
        tLabels(ind,:) = i; %[690  * 1]
    end
    for m = 1:7
        ind1 = 14 + m;
        ind2 = (i-1)*7 + m;
        testing_faces(:,ind2) = illum(:,ind1,i);  
        tstLabels(ind2,:) = i;  %[204 * 1]
    end
end
tFeatures = trainning_faces';
tstFeatures = testing_faces';
