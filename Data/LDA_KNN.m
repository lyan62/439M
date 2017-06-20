%% get mu for each class (200 class in total)
function [class,accuracy] = LDA_KNN(k,tFeatures,tstFeatures,tLabels, tstLabels)
C_num =length(unique(tLabels));
N = size(tFeatures,1)/C_num;
d = size(tFeatures,2);
SW = zeros(d,d);
SB = zeros(d,d);

% total mean of all samples
total_mu = transpose(sum(tFeatures,1)/size(tFeatures,1)); %[504 * 1]

for i  = 1:C_num   %for each class
    c = tFeatures((i-1)*N+1: (i-1)*N+N,:)';  %[504*2]
    mu = mean(c,2); %[504 * 1]
    minus_mu = c - repmat(mu,1,N);
    si{i} = minus_mu * minus_mu';
    SW= SW + si{i};
    SBi{i} = N*(mu - total_mu)*(mu - total_mu)';
    SB = SB + SBi{i};
end

% use  SVD for solving eigenvectors
[U D V] = svd(pinv(SW)*SB); 
W = V(:,1:C_num-1); %[504*199](use C_num-1 to project it to a c-1 dimension)
%[W,EV] = eigs(pinv(SW)*SB, C_num-1);

Yt = W'*tFeatures';    %[199*400]
Ytst = W'*tstFeatures'; %[100*200]
    
%% classify using KNN
% data dimension used for KNN is :
% testdata(200 * ..), trainning data(400 * ..)
[class,accuracy] = KNNEval(k,Yt', tLabels, Ytst', tstLabels);
end
% accuracy(NN) : 0.61 ; (3NN) 0.62

