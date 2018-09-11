function idx = MCLS(fea, knear)

% Inputs:
%   fea:struct data set
%       data : nSample * nFeature
%       target : nLabel *  nSample 
%   knear:Number of neighbors
%
% Output:
%   idx:Feature sorting by score
%
% Example£º
%    fea = load('sample.mat');
%    knear = 5; 
%    idx = MCLS(fea,knear);
%
% Reference:
% Rui Huang, Weidong Jiang, Guangling Sun, Manifold-based constraint Laplacian score for multi-label feature selection, 
% Pattern Recognition Letters, 112 (2018) 346-352.
% Copyright: Rui Huang(huangr@shu.edu.cn Shanghai University)   Weidong Jiang(jiangweidong@shu.edu.cn  Shanghai University)


[m,n] = size(fea.data);

W = full(constructW_PKN(fea.data',knear,1));%Calculating affinity matrix
Wada = estimate_top_struct(fea.data,knear);
MU = build_label_manifold(fea.target',Wada,1);
W_labels = full(constructW_PKN( NormalizeFea(MU)',knear,1));%Label adjacency matrix
W_new = W.*W_labels; % Adjacency matrix after multiplication

conslapscore = LaplacianScore(NormalizeFea(fea.data), W_new); % Laplacian score

options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Binary';
options.t = 1;
options.k = knear;
Wones = full(constructW(fea.data,options)); %Calculating SM matrix
Wzeros = ones(m,m)-Wones;

DM = diag(sum(Wones,2));
LM = DM - Wones;
DC = diag(sum(Wzeros,2));
LC = DC - Wzeros;
for i = 1 : n
    Cr(i,:) = (fea.data(:,i)'*LM*fea.data(:,i))/(fea.data(:,i)'*LC*fea.data(:,i));
end

lapsum = conslapscore.*Cr;

[~,idx] = sort(-lapsum);
idx = idx';

end
        
        
        



