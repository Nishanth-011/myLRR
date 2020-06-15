%%%%%测试半监督学习的效果
%%%%根据需求使用数据集和算法
%%%%rate7是最后的结果
clear;
for time = 1:7
load COIL20;
% load ORL_32x32;
%load YaleBext_3232
%%%%%%%%%%%%%
% load umist
% fea = X';
%%%%%%%%%%%%%%%
folder_now = pwd;
addpath([folder_now, '\funs']);
samp_num = size(fea,1);
nnClass = length(unique(gnd));  % The number of classes;
num_Class=[];
for i=1:nnClass
  num_Class=[num_Class length(find(gnd==i))]; %The number of samples of each class
end

fea =  NormalizeFea(fea);
test = fea;
runtimes = 10;
gama_1 = 50;
gama_2 = 11;
sele = time;
minU0 = 1e-12;
maxU0 = 1e5;  
    %[A OBJ] = LRSA(test', gama_1, gama_2);%LRRADP
    %[A OBJ] = solve_lrr(test', 1);%LRR
    [A OBJ] =  LRRHWAP(test', 50, 11,5,64);
    A = NormalizeFea(A); 
    A = A + 0.0000001*ones(size(A));
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
        F = (D+U0-W+Umin)\U0*Y;
        [maxF, idF] = max(F,[],2);
        for j = 1:samp_num
            FF(j,idF(j)) = 1;
        end
        recogNum = sum(sum((cLab.*FF).*TestF));
        testNum = samp_num-sele*nnClass;
        ratio = double(recogNum)/testNum;
        rate(r) = ratio;
    end
    %max(rate)
    rate7(time) = mean(rate)
    std(rate)
end
%---------------------------------------------------------------