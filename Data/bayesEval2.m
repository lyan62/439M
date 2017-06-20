function [class,accuracy] = bayesEval2(tFeatures, tLabels, tstFeatures, tstLabels)
%% get MEL(mu_est & cov_est )for each class (200 class in total)
C_num =length(unique(tLabels));
N = size(tFeatures,1)/C_num;  % N = 2 samples in each class
d = size(tFeatures,2);  % 504

delta = 1;
for i  = 1:C_num   %for each class
    p(i) = 1/C_num; 
    c = tFeatures((i-1)*N+1: (i-1)*N+N,:)';  %[504*2]
    est_mu{i} = mean(c,2); %[504 * 1]
    minus_mu = c - repmat(est_mu{i},1,N);%[504 * 2]
    est_var{i} = (minus_mu * minus_mu')/N + delta *eye(d);
    inv_var{i} = inv(est_var{i});
    W{i} = -(inv_var{i})/2;
    w{i} = inv_var{i} * est_mu{i};
    w0{i} = (-est_mu{i}'*inv_var{i}*est_mu{i})/2 - log(det(est_var{i}))/2 + log(p(i));
end

% bayes boundary
count = 0;
num_tst = size(tstFeatures,1);
tstFeatures = tstFeatures';
class = zeros(C_num,1);
for t = 1:num_tst
    tst = tstFeatures(:,t);
    for c  = 1 : C_num
        determinant(c) = tst'*W{c}*tst + w{c}'*tst + w0{c};
    end
    [M,I] = max(determinant);
    class(t) = I;
    if class(t) == tstLabels(t)
        count = count + 1;
    end
end

accuracy = count / num_tst;
display(accuracy);
end

