clear;
clc;
%%
load 'pose.mat'
s = size(pose,1)*size(pose,2);
num_class = size(pose,4);
num_t = 4;
num_tst = 9;
for i  = 1: num_class
    for n  = 1:num_t
        ind = (i-1)*num_t + n;
        trainning_faces(:,:,ind) = pose(:,:,n,i);
        tFeatures(ind,:) = reshape(trainning_faces(:,:,ind),s,1);  %[680 * 1920]
        tLabels(ind,:) = i; %[690  * 1]
    end
    for m = 1:num_tst
        ind1 = num_t + m;
        ind2 = (i-1)*num_tst + m;
        testing_faces(:,:,ind2) = pose(:,:,ind1,i);  
        tstFeatures(ind2,:) = reshape(testing_faces(:,:,ind2),s,1); %[204 * 1920]
        tstLabels(ind2,:) = i;  %[204 * 1]
    end
end

%%