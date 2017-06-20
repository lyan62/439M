function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%
% U contains the principal components [n*n]
% S contains a diagonal matrix
% Useful values
[m, n] = size(X);
covMatrix = cov(X);
[U,S,V] = svd(covMatrix);

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%







% =========================================================================

end
