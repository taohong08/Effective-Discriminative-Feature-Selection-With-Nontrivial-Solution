X = rand(150,50);
gnd = [ones(50,1); 2*ones(50,1); 3*ones(50,1)];
lambda = 1;

[ W obj err ] = DFS_L21( X,gnd,lambda);

[~,fearank] = sort(sum(W.*W,2),'descend');
num_of_fea = 10;
selected_fea = fearank(1:num_of_fea);
Xnew = X(:,selected_fea);