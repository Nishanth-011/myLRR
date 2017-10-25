%这个文档主要是产生带有噪声的数据集
clear;
load ORL_32x32;
[m,n] = size(fea);
fea = fea/256;
n = sqrt(n);
datan1 = fea;
datan2 = fea;
for i = 1:m
    data = reshape(fea(i,:),[n,n]);
    data1 = imnoise(data,'salt & pepper',0.1);
    data2 = imnoise(data,'salt & pepper',0.2);
    datan1(i,:) = reshape(data1, [1,n*n]);
    datan2(i,:) = reshape(data2, [1,n*n]); 
end
save('ORL32NO1','datan1');
save('ORL32NO2','datan2');