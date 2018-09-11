function MU = build_label_manifold(Y, W, lambda)
% 
% BUILD_LABEL_MANIFOLD      The label manifold building part of the algorithm ML^2.
% 
% Description
%   MU = BUILD_LABEL_MANIFOLD(X, Y, K, lambda) is the label manifold building part of the algorithm ML^2. 
%   It constructs the label manifold via L quadratic programming problems.
% 
% Inputs:
%   Y: multi-label matrix corresponding to the training samples in X above (N x L). Note that each element 
%   in this matrix can only take 1 or -1, where 1 represents the corresponding label is relevant and -1 represents 
%   the corresponding label is irrelevant.
%   W: weight matrix
%	lambda: parameter in the constraint (3) in our paper.
%     
% Output:
% 	MU: constructed numerical labels.
% 
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

fprintf(1,'Build the label manifold.\n');

[N, L] = size(Y);
M=speye([N,N]);
for i=1:N
   w = W(i,:);
   M(i,:) = M(i,:) - w;
   M(:,i) = M(:,i) - w';
   M = M + w'*w;
end

% For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...
M(isnan(M)) = 0;
M(isinf(M)) = 0;

% Quadratic programming
b = zeros(N,1)-lambda;
options = optimoptions('quadprog',...
    'Display', 'off');
for k=1:L
    A = -diag(Y(:,k));
    MU(:,k) = quadprog(2*M, [], A, b, [], [], [], [],[], options);
end

end