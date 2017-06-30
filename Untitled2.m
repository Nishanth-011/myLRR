load YaleB_3232
h = waitbar(0,'Please wait...');
kk = 3;
number = 64;
fea=fea(1:kk *number,:);
gnd=gnd(1:kk*number);
fea=NormalizeFea(fea);
[new1 b1 dis1] = LRSA(fea',0.1,0.1);
for i = 1:10
    c =  SpectralClustering(new1,kk);
    idx = bestMap(gnd,c); % 匹配
    accuracy1(i) = length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
end
accuracy1=max(accuracy1);
for i=1:191
    [new2 b2 dis2] = LRSA1(fea',0.1,0.1,i);
    [new3 b3 dis3] = LRSA2(fea',0.1,0.1,i);
    for j = 1:5
         c =  SpectralClustering(new2,kk);
         idx=bestMap(gnd,c); % 匹配
         accuracy22(j)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
         c =  SpectralClustering(new3,kk);
         idx=bestMap(gnd,c); % 匹配
         accuracy33(j)=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
    end
    accuracy2(i)=max(accuracy22);
    accuracy3(i)=max(accuracy33);
    waitbar(i/191)
end       
close(h)
k=5;%knn的k
Y = fea';
Yg = fea';
a=fkNN(Yg,k);
[m,n]=size(Y);
WW=zeros(n,n);
 for i=1:n
     aa=a(i,1:k);
     aa(1)=0;
     for j=1:n
         if any(aa==j)
             WW(i,j)=1;
             WW(j,i)=1;
         end
     end
     WW(i,i) = 0;
 end
[new,OBJ] =  sparse_graph_LRR(Yg,WW);
c =  SpectralClustering(new,kk);
idx=bestMap(gnd,c); % 匹配
accuracy= length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量