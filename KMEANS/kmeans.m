
dtime=[];
error=[];
MIhat=[];
for i=1:10
    stime=cputime;
    errnum=0;
    
    load('K:\DM\实验四\hw4data\COIL20\COIL20.mat','fea','gnd');
  str=strcat('K:\DM\实验四\hw4data\COIL20\10Class\',(int2str(i)),'.mat');%下每次数据
load(str,'sampleIdx','zeroIdx');

fea = fea(sampleIdx,:); % 使用testdata集里的数据
gnd = gnd(sampleIdx,:);
fea(:,zeroIdx) = [];
    
idx =kmeans(fea, 10);% 使用matlab自带的kmeans进行分类
idx=bestMap(gnd,idx); % 匹配
accuracy=length(find(gnd == idx))/length(gnd);% 找到gnd和已经分类号的idx中相匹配的值并计算总量
error(i)=1-accuracy;%找到每次的错误率
MIhat(i)= MutualInfo(gnd,idx);% 利用给出的公式求出判别矩阵和原来分类之间的相似度。
etime=cputime;
dtime(i)=etime-stime;            % 计算总时间

end

