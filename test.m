clear;
%load COIL20;
load YaleB_3232
kk = 3;
number = 64;
fea=fea(1:kk *number,:);
gnd=gnd(1:kk*number);
fea=NormalizeFea(fea);
[new1 b1 dis1] = LRSA(fea');
[new2 b2 dis2] = LRSA1(fea',0.1,0.1,2);
[new3 b3 dis3] = LRSA2(fea',0.1,0.1,2);
k=2;%knn的k
Y = fea';
Yg = fea';
a=fkNN(Yg,k);
b=constractmap(a);
c = transmit(b,0);
d = (c+c')/2;
[new,OBJ] =  sparse_graph_LRR(Yg,d);
for i = 1:10
    %c =  SpectralClustering(new,kk);
    %c =  kmeans(new,kk);
    c =  NJW(new,kk);
    idx=bestMap(gnd,c); % 匹配
    accuracy(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    %c =  SpectralClustering(new1,kk);
    %c =  kmeans(new1,kk);
    c =  NJW(new1,kk);
    idx = bestMap(gnd,c); % 匹配
    accuracy1(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    %c =  SpectralClustering(new2,kk);
    %c =  kmeans(new2,kk);
    c =  NJW(new2,kk);
    idx=bestMap(gnd,c); % 匹配
    accuracy2(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    %c =  SpectralClustering(new3,kk);
    %c =  kmeans(new,kk);
    c =  NJW(new3,kk);
    idx=bestMap(gnd,c); % 匹配
    accuracy3(i)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
end
accuracy_=max(accuracy);
accuracy_m=mean(accuracy);
accuracy_t=std(accuracy);
accuracy1_=max(accuracy1);
accuracy1_m=mean(accuracy1);
accuracy1_t=std(accuracy1);
accuracy2_=max(accuracy2);
accuracy2_m=mean(accuracy2);
accuracy2_t=std(accuracy2);
accuracy3_=max(accuracy3);
accuracy3_m=mean(accuracy3);
accuracy3_t=std(accuracy3);