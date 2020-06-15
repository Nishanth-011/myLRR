clear;
addpath('Datasets/');
% load CCC;
load data_Letters
gtlabels = ll;
gnd = zeros(size(gtlabels));
num = 50;
% for i = 1:26
%     gnd(find(gtlabels == char('A'-1+i))) = i;
%     gnd(find(gtlabels == char('a'-1+i))) = i+26;
% end
gnd = ll;
new = [];
newgnd = [];
for i =1:26
    new0 = fea(find(gnd ==i),:);
    newgnd0 = gnd(find(gnd ==i));
    if num < length(newgnd0)
        new = [new;new0(1:num,:)];
        newgnd = [newgnd;newgnd0(1:num)];
    else
        new = [new;new0(1:length(newgnd0),:)];
        newgnd = [newgnd;newgnd0(1:length(newgnd0))];
    end
end
fea = new;
gnd = newgnd;
save(strcat('Datasets/Letters',num2str(num)),'fea','gnd')