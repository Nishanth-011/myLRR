%这个文档主要是产生带有噪声的数据集
clear;
load COIL20;
[m,n] = size(fea);
%fea = fea/256;
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
fea = datan1;
save('COIL20NO1','fea','gnd');
fea = datan2;
save('COIL20NO2','fea','gnd');