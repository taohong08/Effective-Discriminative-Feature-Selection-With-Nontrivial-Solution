function [ W obj err ] = DFS_L2p( data,gnd,lambda,p, options, maxIter, W0 )
% Input:
%   data - Data matrix. Each row vector of fea is a data point.
%   gnd - Colunm vector of the label information for each data point.
%   lambda - trade-off parameter.                    
%   options - parameters for LDA
%   maxIter - Number of iteration.
% Output:
%   W - Transformation matrix.
%   obj - Colunm vector to record the objective value in each iteration

MAX_MATRIX_SIZE = 4500; % You can change this number according your machine computational power
EIGVECTOR_RATIO = 0.1; % You can change this number according your machine computational power

if (~exist('options','var'))
    options = [];
end

if ~isfield(options,'ReguAlpha')
    options.ReguAlpha = 0.1;
end
if ~isfield(options,'eps')
    options.eps = 1e-6;
end
if ~exist('maxIter','var')
    maxIter = 20;
end
if ~exist('p','var')
    p = 1;
end

[nSmp, nFea] = size(data);
obj = zeros(maxIter,1);
err = zeros(maxIter,1);
if (~exist('W0','var'))
    d = ones(nFea,1);
else
    d = sqrt(sum(W0.*W0,2)+ options.eps).^(p-2)   ;
    d = p/2*d;
end
[nSmp,nFea] = size(data);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

% calculate the between-class scater matrix
Hb = zeros(nClass,nFea);
for i = 1:nClass,
    index = find(gnd==classLabel(i));
    classMean = mean(data(index,:),1);
    Hb (i,:) = sqrt(length(index))*classMean;
end
WPrime = Hb'*Hb;
WPrime = max(WPrime,WPrime');

dimMatrix = size(WPrime,2);
if Dim > dimMatrix
    Dim = dimMatrix;
end

% calculate the total scate matrix
DPrime = data'*data;
for i=1:size(DPrime,1)
    DPrime(i,i) = DPrime(i,i) + options.ReguAlpha;
end
DPrime = max(DPrime,DPrime');

if isfield(options,'bEigs')
    bEigs = options.bEigs;
else
    if (dimMatrix > MAX_MATRIX_SIZE) && (Dim < dimMatrix*EIGVECTOR_RATIO)
        bEigs = 1;
    else
        bEigs = 0;
    end
end


dtemp = d;
for iter = 1:maxIter
    D = spdiags(d,0,nFea,nFea);
    % solve the generalized eigen-problem
    % transform the original minimization problem to a maximization problem
    % max_{W'*St*W =I} {-tr{W'*(Sb - lambda*D)*W}}
    WWPrime = WPrime - lambda * D;
    if bEigs
        %disp('use eigs to speed up!');
        option = struct('disp',0);
        W = eigs(WWPrime,DPrime,Dim,'la',option);
    else
        [W, eigvalue] = eig(WWPrime,DPrime);
        eigvalue = diag(eigvalue);
        [~, index] = sort(-eigvalue);
        %         [junk, index] = sort(eigvalue,'descend');
        W = W(:,index);
        
        if Dim < size(W,2)
            W = W(:, 1:Dim);
        end
    end
   
    d = sqrt(sum(W.*W,2)+ options.eps)  ;
    err(iter) = sum(abs(d-dtemp));
    dtemp = d;
    d = d.^(p-2);
    d = (0.5*p)*d;
    temp = W'*WPrime*W;
    obj(iter) = -trace(temp) + lambda*sum(sqrt(sum(W.*W,2)).^p);
    if mod(iter,5) == 0
        fprintf(' %d iterations finished!\n', iter);
    end
end



end

