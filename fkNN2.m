function [R,D] = fkNN2(X,k,distanc,target)
%这个代码主要用于找最近邻每一列是一个元素
%输入参数：
%X--原始的数据矩阵
%k--保留的个数，默认为2
%distanc--是指使用的距离，0为欧式距离，1为余弦距离，默认为0
%target--选择最大或者最小，0为最小1为最大，默认为0
%输出：
if nargin < 4
    target = 0;
end
if nargin < 3
    distanc = 0;
end
if nargin < 2
    k = 2;
end
saveX = X;
[m,n] = size(saveX);
distanceX = zeros(n,n);
result = zeros(n,k);
dis = zeros(n,k);
for i =1:n
    saveX = repmat(X(:,i),1,n);
    saveX = saveX-X;
    saveX = saveX.^2;
    new = sum(saveX);
    distanceX(i,:) = new;
end
[a,b] = sort(distanceX,2);
result = b(:,2:k+1);
dis = a(:,2:k+1);
R = result;
D = dis;