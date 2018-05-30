%k近邻函数im是数据集，k是近邻数,返回一个n*k+1的矩阵每一行前k个是近邻的标号，k+1个是第k个的距离
function Z = fkN(im,k)
[m,n] = size(im);
save=[];
result=zeros(n,k);
for i=1:n
    save=im(:,i);
    for l=1:k
        [y,num]=min(save);
        result(i,l)=num;
        result(i,l+1)=y;
        save(num)=1000;
    end
end
Z=result;