function  X = isomap(X, k, d)
%% minifold learning: isometric feacture mapping 
% k : neighbors
% x : data set 
% d : low dimension;
if nargin < 1
    rand('state',123456789)
    N=1000;
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
    k = 30;
    d = 2;
end
% step1: Calculate the k nearest distance 
[m, ~] = size(X); %d<~
D = zeros(m);
for i =1 : m
    xx = repmat(X(i, :), m, 1);
    diff = xx - X;
    dist = sum(diff.* diff, 2);
    [dd, pos] = sort(dist);
    index = pos(1 : k + 1)';
    index2 = pos(k + 2 : m);
    D(i,index) = sqrt(dd(index));
    D(i, index2) = inf;
end
%step2: recalculate shortest distant matrix
 
for k=1:m
    for i=1:m
        for j=1:m
            if D(i,j)>D(i,k)+D(k,j)
                D(i,j)=D(i,k)+D(k,j);
            end
        end
    end
end
% for i = 1 : m
%     for j = 1 : m
%         if D(i,j) == inf
%             D(i,j) = D(j,i);
%         end
%     end
% end
% step3: adapt MDS algorithm to reduce dimension
Z = MDS(D, d);
% show the distribution
figure(2)
scatter(Z(1, :), Z(2, :), point_size, tt, 'filled');
grid off; axis off; 


function Z = MDS(D, d)
%% Multiple Dimensional Scailing Method 
% Suppose there exists n samples in m-dim, try to reduce dimension
% D: distence matric 
% d: dimension
% Z: new sample \in R^(d*n)
[n1, n2] = size(D);
if n1~=n2
    printf('D is not a square matrix');
else 
    n = n1;
    e = ones(n,1);
    H = eye(n) - 1 / n * (e *e');
    D2 = D .* D;
    B =-1/2 * H' * D2 * H;
    [eigenvector, eigenvalue] = eig(B);
    eigenvalue = diag(eigenvalue);
    [~, pos] = sort(eigenvalue,'descend');
    index = pos(1:d);
    eigenvector = eigenvector(:, index);
    Z = diag(eigenvalue(index))^(0.5) * eigenvector';
end