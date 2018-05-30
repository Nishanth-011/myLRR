function U=NJW(data,k)
%%NJW算法  选取拉氏矩阵的前K个最大特征值对应的特征向量，使其在R（k）空间中构成与原数据一一对应的表述，并在该空间内进行聚类
%%data n*m n样本个数  m 维度
%% k  选择前k个最大特征值对应的特征向量

%计算相似矩阵
affinity = AffinityMatrix(data);
% 计算对角矩阵
D=eye(size(affinity,1));
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end

%计算拉普拉斯矩阵，采用非规范化矩阵
% L=D-affinity;

% 规范化
for i=1:size(affinity,1)
    for j=1:size(affinity,2)
        L(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
    end
end


%计算特征值特征向量
[eigVectors,eigValues] = eig(L);

% 选取前K个最大特征值
[eigValues, ind] = sort(diag(eigValues), 'descend');

nEigVec =eigVectors(:,ind(1:k));

% 构造归一化矩阵U从获得的特征向量
U=zeros(size(nEigVec,1),k);
for i=1:size(nEigVec,1)
    n = sqrt(sum(nEigVec(i,:).^2)); 
    if n == 0
         U(i,:) = nEigVec(i,:);
    else
        U(i,:) = nEigVec(i,:) ./ n;
    end
end
U = kmeans(U,k);
end

