function [x ft] = EProjSimplex_fast(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%
%% use matrix to boost the speed

if nargin < 2
    k = 1;
end


n = size(v,1);
ft=ones(n,1);
v0 = v-mean(v,2)*ones(1,n) + k/n;
%vmax = max(v0);
vmin = min(v0,[],2);
place = find(vmin<0);
place2 = find(vmin>=0);
v1 = zeros(n,n);
if length(place) > 0
    f = ones(n,1);
    lambda_m = zeros(n,1);
    while length(place)>0 && sum(abs(f(place))) > 10^-10*length(f) 
        v1(place,:) = v0(place,:) - lambda_m(place);
        posidx = v1>0;
        npos = sum(posidx(place,:),2);
        g = -npos;
        f = sum(v1.*posidx,2) - k;
        lambda_m = lambda_m - f/g;
        ft(place)=ft(place)+1;
        if length(find(ft>100))>0
            x(find(ft>100),:) = max(v1(find(ft>100),:),0);
            place = setdiff(place, find(ft>100));
        end
    end
    x = max(v1,0);
end
if length(place2>0)
    x(place2,:) = v0(place2,:);
end
end