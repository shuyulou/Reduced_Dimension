function X = LLE(X, k, d)
%% minifold learning: locally linear embedding
% k : neighbors
% x : data set 
% d : low dimension;
if nargin < 1
    N = 2000;
    rand('state',123456789);
    %Gaussian noise
    noise = 0.001*randn(1,N);
    %standard swiss roll data
    tt = (3*pi/2)*(1+2*rand(1,N));   height = 21*rand(1,N);
    X = [(tt+ noise).*cos(tt); height; (tt+ noise).*sin(tt)];
    
    %show the picture
    point_size = 20;
    figure(1)
    cla
    scatter3(X(1,:),X(2,:),X(3,:), point_size,tt,'filled');
    view([12 12]); grid off; axis off; hold on;
    axis off;
    axis equal;
    drawnow;
    X = X';
    k = 12;
    d = 2;
end
[m, ~] = size(X); % d < ~
lambda = 1e-10;
W = zeros(m); 
e = ones(k,1);
for i =1 : m
    xx = repmat(X(i, :), m, 1);
    diff = xx - X;
    dist = sum(diff.* diff, 2);
    [~, pos] = sort(dist);
    index = pos(1 : k + 1)';
    index(index == i) = [];
    w_numerator = (X(index, :) * X(index, :)' + lambda * eye(k)) \ e;
    w_denominator = e' * w_numerator;
    w = w_numerator / w_denominator;
    W(i, index) = w;
end
W =sparse(W);
I = eye(m);
A = (I - W)' * (I - W);
[eigenvector, eigenvalue] = eig(A);
eigenvalue = diag(eigenvalue);
[~,pos] = sort(eigenvalue);
index = pos(1: d+1);
tran = eigenvector(:, index);
p =sum(tran.*tran);
j = find(p == min(p));
tran(:, j) = [];
X = tran;
figure(2)
scatter(X(:,1),X(:,2),point_size, tt,'filled');
grid off; axis off;