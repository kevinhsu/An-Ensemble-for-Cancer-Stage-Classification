function [new_targets, count] = targetsPrepocessing (targets, method, para)
%
%   Preprocessing Method of Targets:
%
%   1: N Equal divisions -- 10, 20, 30, 40, 50 ... para is N classes
%   2; Logarithm division -- 2, 4, 8, 16, 32 ...para is 2
%   3: Multiple-increasing division -- 12, 36, 72, 120 ... para is 12
%
%
if method == 1
    [m, n] = size(targets);
    new_targets = zeros(m, 1);
    count = zeros(para,1);
    maxTargets = max(targets);
    minTargets = min(targets);    
    for j = 1:m
        for i = 1:para
            if i == 1
                if minTargets + (maxTargets - minTargets) * ((i - 1) ./ para) <= targets(j) && targets(j) <= minTargets + (maxTargets - minTargets) * (i ./ para)
                    new_targets(j) = i;
                    count(i) = count(i) + 1;
                end
            else
                if minTargets + (maxTargets - minTargets) * ((i - 1) ./ para) < targets(j) && targets(j) <= minTargets + (maxTargets - minTargets) * (i ./ para)
                    new_targets(j) = i;
                    count(i) = count(i) + 1;
                end
            end
        end
    end
end

if method == 2
    [m, n] = size(targets);
    N = ceil(log(max(targets)) ./ log(para));
    temp_count = zeros(N, 1);
    new_targets = zeros(m, 1);
    for i = 1:N
        for j = 1:m
            if i == 1
                if targets(j) <= power(para, i)
                    new_targets(j) = i; 
                    temp_count(i) = temp_count(i) + 1;
                end
            else
                if (targets(j) > power(para, i - 1)) && (targets(j) <= power(para, i))
                    new_targets(j) = i; 
                    temp_count(i) = temp_count(i) + 1;
                end
            end
        end
    end
    count = temp_count;
end
    
if method == 3
    para = 12;
    N = ceil((-1 + sqrt(1 + 4 * (max(targets) * 2 / para))) / 2);
    [m, n] = size(targets); 
    temp_count = zeros(N ,1);
    new_targets = zeros(m, 1);
    for j = 1:m
        for i = 1:N
            if i == 1
                if targets(j) <= para * (i+1) * i / 2
                    new_targets(j) = i; 
                    temp_count(i) = temp_count(i) + 1;
                end
            else
                if (targets(j) > para * (i-1) * i / 2 ) && (targets(j) <= para * (i+1) * i / 2 )
                    new_targets(j) = i; 
                    temp_count(i) = temp_count(i) + 1;
                end
            end
        end
    end
    count = temp_count;
end
b = bar(count);
ch = get(b);



