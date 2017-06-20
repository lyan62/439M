function [class,accuracy] = PCA_bayes2(num,tFeatures,tstFeatures,tLabels, tstLabels)
features = [tFeatures; tstFeatures]; %[600 * 504]
features = features'; % need [m*n] for applyPCA
projected = applyPCA(features); %  Y = [504*600]
projected = projected'; %[600 * 504]
% reduce dimension to 200(use first 200 principal components)
num_tfeatures= size(tFeatures,1);
% reduce dimension to num(use first (num) principal components)
projected_t = projected(1:num_tfeatures,1:num);
projected_tst = projected(num_tfeatures + 1:end,1:num);
[class,accuracy] = bayesEval2(projected_t, tLabels, projected_tst, tstLabels);
end