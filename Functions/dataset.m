r = 10; %内径
w = 6;  %   外径= r+w
num = 2000; %数据点总数
RandStream.setDefaultStream ...
     (RandStream('mt19937ar','seed',sum(100*clock))); %推荐用这个
dR = rand(1,num)*w;
dThet = rand(1,num)*2*pi;
X = (r+dR) .* cos(dThet);
Y = (r+dR).* sin(dThet);
%原始数据
data = [X;Y];
dataTargets= myHardlim(Y); %目标分类 logical 型
dataTargets = dataTargets .* 1.0;% logical 型 转换为 double型
flag1 = find(dataTargets== 1) ;
plot(data(1,flag1),data(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data(1,flag0),data(2,flag0),'b.','MarkerSize',5);
title('原始数据')
%移动后的数据 data1
dv= 1;  % 两个“半月”的垂直距离 ，正数表示离得更远
dh = 10 ;   %两个“半月”的水平距离 ，正数表示向右 离得更远
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data1 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data1(1,flag1),data1(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data1(1,flag0),data1(2,flag0),'b.','MarkerSize',5);
title('移动 dv=1,dh=10 后的数据 ')
%移动后的数据 data2
dv= -4;  % 两个“半月”的垂直距离 ，正数表示离得更远
dh = 10 ;   %两个“半月”的水平距离 ，正数表示向右 离得更远
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data2 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data2(1,flag1),data2(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data2(1,flag0),data2(2,flag0),'b.','MarkerSize',5);
title('移动 dv=-4,dh=10 后的数据 ');
%移动后的数据 data3
dv= -5;  % 两个“半月”的垂直距离 ，正数表示离得更远
dh = 10 ;   %两个“半月”的水平距离 ，正数表示向右 离得更远
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data3 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data3(1,flag1),data3(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data3(1,flag0),data3(2,flag0),'b.','MarkerSize',5);
title('移动 dv=-5,dh=10 后的数据 ');
%移动后的数据 data4
dv= -6;  % 两个“半月”的垂直距离 ，正数表示离得更远
dh = 10 ;   %两个“半月”的水平距离 ，正数表示向右 离得更远
X = data(1,:);
Y = data(2,:);
flag = find( Y < 0 );
Y(1,flag) = Y(1,flag) - dv;
X(1,flag) = X(1,flag) + dh;
data4 = [X;Y];
figure;
flag1 = find(dataTargets== 1) ;
plot(data4(1,flag1),data4(2,flag1),'r.','MarkerSize',5);
hold on;
flag0 = find(dataTargets== 0) ;
plot(data4(1,flag0),data4(2,flag0),'b.','MarkerSize',5);
title('移动 dv=-6,dh=10 后的数据 ');
%保存数据集到文件
currentpath = eval( 'pwd' ); %获取当前工程目录路径
disp('current path : ');
disp(currentpath);
ts = strcat(currentpath,'\shuangyuedata.mat');
save(ts,'data','data1','data2','data3','data4','dataTargets');