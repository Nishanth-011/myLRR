%用来将constractmap获得的最近邻图（二值）转化为近邻传递图（二值）
%输入：
%data——最近邻图（二值）
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
for i = 1:m
    for j = 1:m
        if datasave(i,j) >0
            datasave(i,j) = 1;
        end
    end
end
graph = datasave;