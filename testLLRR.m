%%%%%TEST COIL20%%%%%%%%%
%clc,
clear;
%load COIL20
load YaleB_3232
nn = 20;
fea = fea(1:64*nn,:);
gnd = gnd(1:64*nn);
folder_now = pwd;
addpath([folder_now, '\funs']);
%% reduce demension by PCA
samp_num = size(fea,1);
nnClass = length(unique(gnd));  % The number of classes;
num_Class=[];
for i=1:nnClass
  num_Class=[num_Class length(find(gnd==i))]; %The number of samples of each class
end
fea =  NormalizeFea(fea);
test = fea;
runtimes = 10;
sele = 4;
minU0 = 1e-12;
maxU0 = 1e5;
k=5;%knn的k
Y = fea';
Yg = fea';
a=fkNN(Yg,k);
[m,n]=size(Y);
WW=zeros(n,n);
% ******************************
% 这一段的内容是查找是否在k近邻里面
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
% ****************************
%*********************************
%这一段是使用距离找最近邻
% for i=1:n
%    aa=a(i,k+1);
%    for j=1:n
%        if norm(Yg(:,i)-Yg(:,j))<=aa
%            WW(i,j)=1;
%            WW(j,i)=1;%这个是用来对称的
%        end
%    end
%    WW(i,i)=0;
% end
%**************************************

    [A,OBJ] =  sparse_graph_LRR(Yg,WW);
    A = NormalizeFea(A);
    
    AG = A;
    save('AG','AG');
    save('OBJ','OBJ');
    
    W = A;
    D = diag(sum(A));

    for r=1:runtimes
%---------------------------------------------------------------   

        Y = zeros(samp_num, nnClass);
        cLab = zeros(samp_num, nnClass);
        FF = zeros(samp_num, nnClass);
        TestF = ones(samp_num, nnClass);
        U0 = zeros(samp_num, samp_num);
        Umin = minU0*ones(samp_num, samp_num);

        for  j=1:nnClass
            idx=find(gnd==j);
            cLab(idx, j) = 1;
            randIdx=randperm(num_Class(j)); %randIdx create m random number, m is the size of idx.
            %randIdx = 1:sele;
            Y(idx(randIdx(1:sele)),j) = 1;
            TestF(idx(randIdx(1:sele)),:) = 0;      
            for s = 1:sele
                U0(idx(randIdx(s)),idx(randIdx(s))) = maxU0;
            end                
        end

        F = inv(D+U0-W+Umin)*U0*Y;
        [maxF, idF] = max(F,[],2);
        for j = 1:samp_num
            FF(j,idF(j)) = 1;
        end

        recogNum = sum(sum((cLab.*FF).*TestF));
        testNum = samp_num-sele*nnClass;
        ratio = double(recogNum)/testNum;
        rate(r) = ratio;
    end
    max(rate)
    mean(rate)
    std(rate)
%---------------------------------------------------------------