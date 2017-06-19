function [R,D] = fkNN(X,k,distanc,target)
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
if distanc == 0
	for i = 1:n
		for j = 1:n
			distanceX(i,j) = norm(saveX(:,i)-saveX(:,j));
		end
    end
    if target == 0
        for i = 1:n
            [a,b] = sort(distanceX(i,:));
            result(i,:) = b(2:k+1);
            dis(i,:) = a(2:k+1);
        end 
    else
        result = zeros(n,k);
        for i = 1:n
            [a,b] = sort(distanceX(i,:),'descend');
            result(i,:)=b(1:k);
            dis(i,:) = a(1:k);
        end
    end
else
	for i = 1:n
		for j = 1:n
			distanceX(i,j) = dot(X(:,i),X(:,j))/(norm(X(:,i))*norm(X(:,j)));
		end
    end
    if target == 0
        for i = 1:n
            [a,b] = sort(distanceX(i,:));
            result(i,:) = b(1:k);
            dis(i,:) = a(1:k);
        end 
    else
        result = zeros(n,k);
        for i = 1:n
            [a,b] = sort(distanceX(i,:),'descend');
            result(i,:)=b(2:k+1);
            dis(i,:) = a(2:k+1);
        end
    end
end
R = result;
D = dis;