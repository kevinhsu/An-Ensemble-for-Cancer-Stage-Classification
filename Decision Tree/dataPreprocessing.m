function [new_features, new_targets] = dataPreprocessing(features, missingTolerance, targets, method, para)

%% Initialise imputation

% Combine Complete-case imputation and Modified simple imputation
new_features = featuresPreprocessing(features, missingTolerance);
new_targets = targetsPreprocessing(targets, method, para);
[m, n] = size(new_features);

missingCount = zeros(1, m);
for i = 1:m
    for j =1:n
        if isnan(features(i, j))
            missingCount(i) = missingCount(i) + 1;
        end
    end
end

count = 0;
for i = 1:m
    if missingCount(i) > missingTolerance * n
        new_features(i - count, :) = [];
        new_targets(i - count, :) = [];
        count = count + 1;
    end
end

[m, n] = size(new_features);

for i = 1:n
    for j = 1:n
    	corMartix(i, j) = corr2(new_features(:, i), new_features(:, j));
    end
end

[c_i, d_i] = find(corMartix == 1);
A = [c_i, d_i];
B = [];
[m_a, n_a] = size(A);
for i = 1:m_a
    if A(i, 1) ~= A(i, 2) && new_targets(A(i, 1)) ~= new_targets(A(i, 2))
        B = [B, A(i, 1)];
    end
end

B = sort(B);
[m_b, n_b] = size(B);

count = 0;
for i = 1:m
    for j = 1:m_b
        if i == B(j)
            new_features(i - count, :) = [];
            new_targets(i - count, :) = [];
            count = count + 1;
        end
    end
end

% Randomise the rows of data
z = [new_features, new_targets];
[m, n] = size(new_features);
shuffledArray = z(randperm(size(z,1)), :);
new_features = shuffledArray(:, 1:n);
new_targets = shuffledArray(:, n+1);












%% Solve unbalanced dataset 

% method = 1 & para = 2
unic = unique(new_targets);
[m, n] = size(unic);

present = zeros(m,1);

for i = 1:m
     [p1, p2] = size(find(new_targets == unic(i)));
     present(i) = max(p1, p2);
end

[featuresMean, featuresVar, featuresMedian, featuresMode, featuresType, missingCount] = statisticalFeatures(new_features);

unbalanced_present = zeros(m,1);
for i = 1:m
    
    unbalanced_present(i) = max(present) - present(i);
    temp_index = find(new_targets == unic(i));
    
    for j = 1:unbalanced_present(i)
        
        temp_rand = round(present(i) * rand(1,1));
        
        if temp_rand == 0
            temp_rand = max(present(i));
        end
            
            new_features = cat(1, new_features, new_features(temp_index(temp_rand), :));
            [m_, n_] = size(new_features);
            
            for i_ = 1:n
                if featuresType(i_) == 2
                    new_feature(m_, i_) =  new_feature(m_, i_) + featuresVar(i_) * rand(1,1);
                end
            end        
            
            new_targets = cat(1, new_targets, new_targets(temp_index(temp_rand), :));
            
    end
end

%% featuresPreprocessing (features, missingTolerance)
function [new_features] = featuresPreprocessing (features, missingTolerance)

% missingTolerance is a percentage value belonging to (0, 1)

[m, n] = size(features);
new_features = zeros(m, n);
[featuresMean, featuresVar, featuresMedian, featuresMode, featuresType, missingCount] = statisticalFeatures(features);

for j = 1:n
    for i = 1:m
        if isnan(features(i, j))
            if featuresType(j) == 1
                new_features(i, j) = featuresMedian(j);
            end
            if featuresType(j) == 2
                new_features(i, j) = featuresMean(j);
            end
        else
            new_features(i, j) = features(i, j);
        end
    end
end

count = 0;
for j = 1:n
    if missingCount(j) > missingTolerance * m
        new_features(:, j - count) = [];
        count = count + 1;
    end
end

% Principle Components Analysis
% [W, pc, latent, tsquare] = princomp(new_features);
% new_features = [];
% contribution = 0;
% totalLatent = sum(latent);
% i = 1;
% while contribution ./ totalLatent <= 0.999999
%     contribution = contribution + latent(i);
%     new_features(:, i) = pc(:, i);
%     i = i + 1;
% end
        
% statisticalFeatures
function [featuresMean, featuresVar, featuresMedian, featuresMode, featuresType, missingCount] = statisticalFeatures(features)

[m, n] = size(features);
missingCount = zeros(1, n);
featuresMean = zeros(1, n);
featuresVar = zeros(1, n);
featuresMedian = zeros(1, n);
featuresMode = zeros(1, n);

for j = 1:n
    k = 1;
    featureColumn = [];
    for i = 1:m
        if isnan(features(i, j))
            missingCount(j) = missingCount(j) + 1;
        else
            featureColumn(k) = features(i, j);
            k = k + 1;
        end
    end
    
    featuresMean(j) = mean(featureColumn);
    featuresVar(j) = var(featureColumn);
    featuresMedian(j) = median(featureColumn);
    featuresMode(j) = mode(featureColumn);
% Distinguish numerical and categorical features
    [m_c, n_c] = size(featureColumn);
    featuresType(j) = 1; %numerical

    if featuresType(j) == 1
        for i = 1:n_c
            if featureColumn(i) ~= fix(featureColumn(i))
                featuresType(j) = 2; %ordinary
                break;
            end
        end
    end
end

missingCount = missingCount';
featuresMean = featuresMean';
featuresVar = featuresVar';
featuresMedian = featuresMedian';
featuresMode = featuresMode';
featuresType = featuresType';

%% targetsPrepocessing (targets, method, para)
function [new_targets, count] = targetsPreprocessing (targets, method, para)
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
% b = bar(count);
% ch = get(b);
