%将fkNN计算出来的近邻矩阵和距离转化为每一个元素和到任意元素的距离矩阵
%输入：
%data1——近邻序号矩阵
%data2——近邻序号矩阵对应的距离矩阵
%max——不相邻的元素的距离，默认为1000
function result = mapdis(data1,data2,max)
if nargin < 3
    max = 1000;
end
[m,n] = size(data1);
savedata = ones(m,m)*max;
for i = 1:m
    savedata(i,i) = 0;
end
for i = 1:m
    for j = 1:n
		savedata(i,data1(i,j)) = data2(i,j);
    end
end
result = savedata;
 