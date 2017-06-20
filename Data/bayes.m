function [class,accuracy] = bayes(tFeatures, tLabels, tstFeatures, tstLabels)
%% get MEL(mu_est & cov_est )for each class (200 class in total)
C_num =length(unique(tLabels));
N = size(tFeatures,1)/C_num;  % N = 2 samples in each class
d = size(tFeatures,2);  % 504

delta = 1;
for i  = 1:C_num   %for each class
    c = tFeatures(2*i-1: 2*i,:)';  %[504*2]
    est_mu{i} = mean(c,2); %[504 * 1]
    minus_mu = c - repmat(est_mu{i},1,N);%[504 * 2]
    est_var{i} = (minus_mu * minus_mu')/N + delta *eye(d);
end

% bayes boundary
count = 0;
num_tst = size(tstFeatures,1);
for t = 1:num_tst
    tst = tstFeatures(t,:)';
    for c  = 1 : C_num
        p(c) = 1/C_num; 
        A = -((tst - est_mu{c})' * inv(est_var{i}) * (tst - est_mu{c}))/2;
        B = - trace(logm(est_var{c}))/2;
        %B = - log(det(est_var{c}))/2;
        %C = log(p(c)); same for all classes, can be ignored
        determinant(c) = A + B ; %+ C;
    end
    [M,I] = max(determinant);
    class(t) = I;
    if class(t) == tstLabels(t)
        count = count + 1;
    end
end

accuracy = count / num_tst;
display(accuracy);

