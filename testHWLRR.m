clc
clear
addpath('Functions/')
clear;
addpath('Datasets/');
addpath('Functions/');
addpath('Functions/Measures/');
addpath('Functions/Measures/ami');
addpath('Functions/Measures/munkres');
addpath('PR_code_AWNLRR/')
addpath('PR_code_AWNLRR/Ncut_9')
addpath('ours');
addpath('D:/mycode/myLRR/Datasets/')
% addpath('LRR/')
% addpath('latlrr/')
dataset_name = 'YaleB_3232';
load(strcat(dataset_name,'.mat'))
parallel = 0;%这个用来控制是否使用并行
switch dataset_name
    case {'cars_uni'}
        ttime = 1;
        basic = 3;
        interval = 0;
        fea = X;
        gnd = Y;
    case {'ORL_32x32','ORL_64x64'}
        interval = 5;%这个是去样本数的间隔
        ttime = 8;
        basic = 0;
%         ttime = floor(length(unique(gnd))/interval);
    case {'YaleB_3232'}
        interval = 6;
        basic = -4;
        ttime = 7;
    case {'COIL20', 'coil20_64x64','Umist'}
        basic = 2;
        interval = 2;%这个是去样本数的间隔
        ttime = 9;
    case {'Yale_32x32', 'Yale_64x64'}
        interval = 15;%这个是去样本数的间隔
        ttime = 1;
    case {'CCC40','CCC50','CCC80','CCC100'}
        interval = 4;%这个是去样本数的间隔
        ttime = 13;
    case {'PD100','PD200'}
        interval = 2;%这个是去样本数的间隔
        ttime = 5;
    case 'lights_27'
        interval =2;%这个是去样本数的间隔
        ttime = 13;
    case 'mnist_all'
        interval = 2;%这个是去样本数的间隔
        num = 200;
        fea = double([train0(1:num,:);train1(1:num,:);train2(1:num,:);train3(1:num,:);train4(1:num,:);train5(1:num,:);train6(1:num,:);train7(1:num,:);train8(1:num,:);train9(1:num,:)]);
        gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
        ttime = floor(length(unique(gnd))/interval);
    case 'USPSfu'
        dnum =50;
        num = dnum;
        fea = [data0(:,1:dnum)';data1(:,1:dnum)';data2(:,1:dnum)';data3(:,1:dnum)';data4(:,1:dnum)';data5(:,1:dnum)';data6(:,1:dnum)';data7(:,1:dnum)';data8(:,1:dnum)';data9(:,1:dnum)'];
        fea = double(fea);
        gnd = double([ones(num,1);2*ones(num,1);3*ones(num,1);4*ones(num,1);5*ones(num,1);6*ones(num,1);7*ones(num,1);8*ones(num,1);9*ones(num,1);10*ones(num,1)]);
        interval = 2;%这个是去样本数的间隔
        ttime = floor(length(unique(gnd))/interval);
end
data = fea;
datal = double(gnd);
% acc = zeros(5,5,5,ttime);
for time1 =1:ttime
    acc = zeros(11,11,11);
    %time_num * ttime - time
    kk = basic +interval*time1;%类别数
    number = find(datal==kk);%每一类的个数
    number=max(number);
    fea=data(1:number,:);%kk *number,:);
    gnd=datal(1:number);%kk*number);
    fea = fea';
    fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
    n = length(gnd);
    nnClass = length(unique(gnd));  

    options = [];
    options.NeighborMode = 'KNN';
    options.k = 10;
    options.WeightMode = 'Binary';
    Z = constructW(fea',options);
    Z = full(Z);
    Z1 = Z-diag(diag(Z));         
    Z = (Z1+Z1')/2;
    DZ= diag(sum(Z));
    LZ = DZ - Z;                
    [F_ini, ~, evs]=eig1(LZ, kk, 0);
    Z_ini = full(Z);
    clear LZ Z Z1 options

    % % if you only have cpu do this 
    % Ctg = inv(fea'*fea+eye(size(fea,2)));

    % % -------- if you have gpu you can accelerate the inverse operation as follows:  ---------- % %
    Xg = gpuArray(single(fea));
    Ctg = inv(Xg'*Xg+eye(n));
    Ctg = double(gather(Ctg));
    clear Xg;
    % lambda1 = 1
    % lambda2 = 0.1
    % lambda3 = 0.20
    % lambda4 = 0.1
    k_num = 3;
    [b,c]=fkNN(fea,k_num);
    aa = constractmap(b);
    aa(find(aa>0))=1;
    % bb = sendk(aa,time,k);
    bb = sendknew(aa,99,k_num);
    % bb1 = sendknew1(aa, time, k);
    [m,~] = size(bb);
    bb(find(bb==0)) =100000;
    bb = bb - diag(diag(bb));
    D = bb;

    for i = 1:11
        for j = 1:11
            for k = 1:11
                % lambda1 = 10
                % lambda2 = 0.001
                % lambda3 = 0.02
                % lambda4 = 0.0001
                lambda1 = 10^(i-6);
                lambda2 = 10^(j-6);
                lambda3 = 10^(k-6);
                miu = 1e-2;
                rho = 1.1;
                max_iter =80;
%                 [Z,S,obj] = AWLRRL21(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
%                 [Z,S,obj] = AWLRRL21(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
%                 [Z,S,obj] = AWLRRf12(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
%                 [Z,S,obj] = AWLRRf12(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
%                 [Z,S,obj] = AWLLRRL1(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
                   [Z,S,obj] = HWLRR(fea,Z_ini,D,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
%                 [Z,S,G,U,obj] = AWLRRnovel(fea,Z_ini,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
                c = kk;
%                 [Z,S,G,U,obj] = AWLRRnovel(fea,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho);
                addpath('Ncut_9');
                Z_out = Z;
                A = Z_out;
                A = A - diag(diag(A));
                A = abs(A);
                A = (A+A')/2;  
                [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,nnClass);
                result_label = zeros(size(fea,2),1);
                for f = 1:nnClass
                    id = find(NcutDiscrete(:,f));
                    result_label(id) = f;
                end
                result = ClusteringMeasure(gnd, result_label);
                acc(i,j,k)  = result(1); 
                kk
                fprintf(strcat(num2str(i),num2str(j),num2str(k),'!',num2str(kk),'!',num2str(result(1))))
            end
        end
    end
    save(strcat('AWLRR/',dataset_name,num2str(kk)),'dataset_name','acc','time1')
end