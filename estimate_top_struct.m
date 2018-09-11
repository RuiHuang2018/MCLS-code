
function W = estimate_top_struct(X, K)
% 
% ESTIMATE_TOP_STRUCT      Estimate the topological structure in the feature space.
% 
% Description
%   W = ESTIMATE_TOP_STRUCT(X, K) estimate the topological structure as ML^2. 
%   It includes two main steps. First, find K nearest neighbors for each training example. 
%   Second, approximate the topological structure of the feature manifold via N standard least square programming problems,
%   where N is the number of training examples. 
% 
% Inputs:
%   X: data matrix with training samples in rows and features in in columns (N x D)
%   K: number of selected nearest neighbors.
%     
% Output:
% 	W: weight matrix
% 
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

fprintf(1,'Estimate the topological structure.\n');

[N,D] = size(X);

neighborhood = knnsearch(X, X, 'K', K+1);
neighborhood = neighborhood(:, 2:end);

if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); 
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

% Least square programming
W = sparse(N, N);
for i=1:N
    neighbors = neighborhood(i,:);
    z = X(neighbors,:)-repmat(X(i,:),K,1); % shift ith pt to origin
    Z = z*z';                                        % local covariance
    Z = Z + eye(K,K)*tol*trace(Z);                   % regularlization (K>D)
    W(i,neighbors) = Z\ones(K,1);                           % solve Zw=1
    W(i,neighbors) = W(i,neighbors)/sum(W(i,neighbors));                  % enforce sum(w)=1
end

end