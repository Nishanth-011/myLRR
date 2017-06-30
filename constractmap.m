%这个函数主要用于将最近领的序号转换为图关系
%输入：fkNN的结果
%输出：方阵，是自己每一行是自己最近邻的就为1，不是的为0
function [result] = constractmap(data)
	[m,n] = size(data);
	savedata = zeros(m,m);
	for i = 1:m
        savedata(i,data(i,:)) = 1;
	end
	result = savedata;