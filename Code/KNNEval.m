function [class,accuracy] = KNNEval(k,tFeatures, tLabels, tstFeatures, tstLabels)
% input: tFeatures: 400 * 59 (for 400 samples)
%        tLables: 400*1 
%        tstFeatures: 200 * 59 (for 200 test samples)
%        tstLables: 200 * 1
%        k: number of neighbors

num_tst = size(tstFeatures,1);
num_t = size(tFeatures,1);
count = 0;
for i  = 1 : num_tst
    for j = 1 : num_t
        diff(j,:) = tFeatures(j,:) - tstFeatures(i,:);
        dist(j) = norm(diff(j,:));
    end
    [Y,I] = sort(dist);
    neighborsInd(i,:) = I(1:k);
    class(i) = calcVote(tLabels(neighborsInd(i,:)));
    if class(i) == tstLabels(i)
        count = count + 1;
    end
end

accuracy = count / num_tst;
display(accuracy);
end