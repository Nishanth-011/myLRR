clear;
im=imread('obj1__0.png');
number=10;%相同数据的个数
Ynew=[];%这是还原后的数据
%I=rgb2gray(im);%转化为灰度
I=im;
k=5;%knn的k
Y=double(I);
%################################
%这一段代码是产生相同数据的个数
%for i=1:number
%   Y(1:600,i)=Y(1:600,1);
%end
%###############################
a = 10;  
b = 4;  
n_gaussian = randn(128, 128);
YY=Y+n_gaussian;
Yg=mapminmax(YY);
a=fkNN(Yg,k);
[m,n]=size(Y);
W=zeros(n,n);
%******************************
%这一段的内容是查找是否在k近邻里面
for i=1:n
    aa=a(i,1:k);
    for j=1:n
        if any(aa==j)
            W(i,j)=1;
            W(j,i)=1;%这个是用来对称的
        end
    end
    W(i,i)=0;
end
%****************************
%*********************************
%这一段是使用距离找最近邻
% for i=1:m
%    aa=a(i,k+1);
%    for j=1:n
%        if norm(Yg(:,i)-Yg(:,j))<=aa
%            W(i,j)=1;
%            W(j,i)=1;%这个是用来对称的
%        end
%    end
%    W(i,i)=0
% end
%n_gaussian = a + b .* randn(1200, 1920);  
[Z, E] = sparse_graph_LRR(YY,W,0.1, 5, 8.0, 2.5, 0);
Ynew=Y*Z+E;