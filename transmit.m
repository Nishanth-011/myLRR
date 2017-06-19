%% transmit: 转换函数，输入每一行是最近邻的关系矩阵
function [graph] = transmit(data)
	[m,~] = size(data);
	datasave = data;
	datan = data;
	for i = 1:m
		for j = 1:m
			for k = 1:m
				if datasave(j,k) > 0
					datasave(j,:) = datasave(j,:) + data(k,:);
				end
			end
		end
	end
	graph = datasave;