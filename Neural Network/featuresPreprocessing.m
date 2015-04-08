function new_features = featuresPreprocessing (features, missingTolerance)

%missingTolerance is a percentage value belonging to (0, 1)

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

%% Principle Components Analysis
[W, pc, latent, tsquare] = princomp(new_features);
new_features = [];
contribution = 0;
totalLatent = sum(latent);
i = 1;
while contribution ./ totalLatent <= 0.999999
    contribution = contribution + latent(i);
    new_features(:, i) = pc(:, i);
    i = i + 1;
end
        
%% statisticalFeatures
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
    featuresType(j) = 1;
%     if max(featureColumn) > 10
%         featuresType(j) = 2;
%     end
    if featuresType(j) == 1
        for i = 1:n_c
            if featureColumn(i) ~= fix(featureColumn(i))
                featuresType(j) = 2;
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




