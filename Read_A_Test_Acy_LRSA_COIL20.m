clc;
clear all;

% load Yale_32x32
% nnClass = length(unique(gnd));
% num_Class=[];
% for i=1:nnClass
%   num_Class=[num_Class length(find(gnd==i))];
% end
load COIL20
nn = 4;
fea=fea(1:72*nn,:);
gnd=gnd(1:72*nn);
%fea=new_data;
% %%
%samp_num = size(fea,1);
nnClass = length(unique(gnd));  % The number of classes;
num_Class=[];
for i=1:nnClass
  num_Class=[num_Class length(find(gnd==i))]; %The number of samples of each class
end

load('AG');

runtimes = 10;
A = AG; 
A = NormalizeFea(A);
samp_num = size(A,1);

fea_num = num_Class(1);
fea_gnd = zeros(fea_num*nnClass, 1);    
for  j=1:nnClass
    fea_gnd((j-1)*fea_num+1:j*fea_num) = j;
end
maxU0 = 1e5;
minU0 = 1e-12;

sele = 10;


for run =1:runtimes
    W = A;
    D = diag(sum(A));

    Y = zeros(samp_num, nnClass);
    cLab = zeros(samp_num, nnClass);
    FF = zeros(samp_num, nnClass);
    TestF = ones(samp_num, nnClass);

    U0 = zeros(samp_num, samp_num);
    Umin = minU0*ones(samp_num, samp_num);
    
    for  j=1:nnClass
        idx=find(fea_gnd==j);
        cLab(idx, j) = 1;
        randIdx=randperm(fea_num); %randIdx create m random number, m is the size of idx.
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

    rate(run) =  ratio;    
end

AC_mean=mean(rate)
AC_std=std(rate)

%AC2 = mean(rate)
