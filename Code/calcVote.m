function class = calcVote(x)
l = length(x);
[a,b]=hist(x,unique(x));
[M,I] = max(a);
class = b(I);
if a(I)<= l/2
    class = x(1);  % if the selected class does not get majority vote, use 1-NN;
%output = sprintf('get class %d with %d vote.',class, a(I));
%disp(output);
end