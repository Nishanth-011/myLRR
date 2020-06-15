%%%%%测试半监督学习的效果
%%%%根据需求使用数据集和算法
%%%%rate7是最后的结果
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Results/newrevise/');
addpath('Datasets/');
addpath('Functions/');
addpath('Functions/Measures/');
addpath('Functions/Measures/ami');
addpath('Functions/Measures/munkres');
addpath('LRR');
addpath('FastESC-master');
addpath('FastESC-master/EBMM_Release');
addpath('SparsifiedKMeans-master');
addpath('SparsifiedKMeans-master/private');
dataset_name = {'ORL_32x32','Umist','YaleB_3232','COIL20','mnist_all','CCC40','Yale_64x64','PD200'};
% load(strcat(dataset_name {data_num},'.mat'))
data_num = 6;
if data_num == 1
load sORL_32x32awlrr
elseif data_num == 2
load sUmistawlrr
elseif data_num == 3
load sYaleB_3232awlrr
elseif data_num == 4
load sCOIL20awlrr
elseif data_num == 5
    load smnist_allawlrr
elseif data_num == 6
    load sCCC40awlrr
elseif data_num == 7
    load sYale_64x64awlrr
elseif data_num == 8
    load sPD100awlrr
end
fea = fea';
folder_now = pwd;
addpath([folder_now, '\funs']);
samp_num = size(fea,1);
nnClass = length(unique(gnd));  % The number of classes;
num_Class=[];
for i=1:nnClass
  num_Class=[num_Class length(find(gnd==i))]; %The number of samples of each class
end
num = 7;
fea =  NormalizeFea(fea);
test = fea;
runtimes = 10;
gama_1 = 50;
gama_2 = 11;
minU0 = 1e-12;
maxU0 = 1e5;
label_r = cell(num,runtimes);
gnd_r = cell(num,runtimes);
rate = zeros(1,runtimes);
A =Z;
%     [A OBJ] = LRSA(test', gama_1, gama_2);%LRRADP
%     [A OBJ] = solve_lrr(test', 1);%LRR
%     [A OBJ] =  LRRHWAP(test', 50, 11,5,99);
    A = NormalizeFea(A); 
    A = A + 0.0000001*ones(size(A));
    AG = A;
    %save('AG','AG');
    %save('OBJ','OBJ');
    W = A;
    D = diag(sum(A));
for time = 1:num
    sele = time;
    AG = A;
    %save('AG','AG');
    %save('OBJ','OBJ');
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
        ll = FF.*TestF;
        ll = (1:nnClass)*ll';
        ll = ll(find(ll>0));
        gndd = cLab.*TestF;
        gndd = (1:nnClass)*gndd';
        gndd = gndd(find(gndd>0));
        label_r{time,r} = ll;
        gnd_r{time,r} = gndd;
        testNum = samp_num-sele*nnClass;
        ratio = double(recogNum)/testNum;
        rate(r) = ratio;
        [aARI(time,r),aAMI(time,r),aNMI(time,r),aACC(time,r)] = evaluate(ll,nnClass,gndd,nnClass);
        aAUC(time,r) = AUC(gndd,ll);
        [aTPR(time,r),aFPR(time,r),aPrecision(time,r),aRecall(time,r),aF1(time,r)] = performanceIndexs(gndd,ll);
    end
    %max(rate)
    rate7(time) = mean(rate);
    amARI = mean(aARI,2);
    amAMI = mean(aAMI,2);
    amNMI = mean(aNMI,2);
    amACC = mean(aACC,2);
    amAUC = mean(aAUC,2);
    amTPR = mean(aTPR,2);
    amFPR = mean(aTPR,2);
    amPrecision = mean(aPrecision,2);
    amRecall = mean(aRecall,2);
    amF1 = mean(aF1,2);
    aaresult = [amARI,amAMI,amNMI,amACC,amAUC,amTPR,amFPR,amPrecision,amRecall,amF1];
%     std(rate);
end
save(strcat('Results/newrevise/',strcat('ss',dataset_name{data_num},'awlrr')))
%---------------------------------------------------------------