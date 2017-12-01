%这个文档主要是产生带有噪声的数据集
clear;
<<<<<<< HEAD
load COIL20;
[m,n] = size(fea);
%fea = fea/256;
=======
<<<<<<< HEAD
load COIL20;
[m,n] = size(fea);
%fea = fea/256;
=======
load ORL_32x32;
[m,n] = size(fea);
fea = fea/256;
>>>>>>> 662380046dce7c0c25aea3f5a3e33db8721bd8d8
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
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
<<<<<<< HEAD
fea = datan1;
save('COIL20NO1','fea','gnd');
fea = datan2;
save('COIL20NO2','fea','gnd');
=======
<<<<<<< HEAD
fea = datan1;
save('COIL20NO1','fea','gnd');
fea = datan2;
save('COIL20NO2','fea','gnd');
=======
save('ORL32NO1','datan1');
save('ORL32NO2','datan2');
>>>>>>> 662380046dce7c0c25aea3f5a3e33db8721bd8d8
>>>>>>> a50b32285e197bde7a9f052ddccc9df1cb606843
