function [corMartix] = Preprocessing(features, missingTolerance, targets, method, para)

%% Initialise imputation & Solve unbalanced dataset 
[new_features, new_targets] = dataPreprocessing(features, missingTolerance, targets, method, para);

%% Imputation based on correlation
M = 1;
[m, n] = size(new_features);


% Until corMartix convergents: norm_2
for iteration = 1:M 
    
    temp_features = new_features;
    temp_targets = new_targets;
    for i = 1:n+1
        for j = 1:n+1
            if i == n+1 && j ~= n+1
                corMartix(i, j) = corr2(new_targets, new_features(:, j));
            end
            if i ~= n+1 && j == n+1
                corMartix(i, j) = corr2(new_features(:, i), new_targets);
            end
            if i == n+1 && j == n+1
                corMartix(i, j) = corr2(new_targets, new_targets);
            end
            if i ~= n+1 && j ~= n+1
                corMartix(i, j) = corr2(new_features(:, i), new_features(:, j));
            end
        end
    end
    
    
    %........
    
end


%% find the index of missing values
function [missing_index] = missingCount(features)

[m, n] = size(features);

missing_index = [];

for i = 1:m
    if isnan(features(i))
        missing_index = cat(1, missing_index, i);
    end
end

missing_index = missing_index';
