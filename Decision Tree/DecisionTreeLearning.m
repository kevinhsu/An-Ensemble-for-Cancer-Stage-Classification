function tree = DecisionTreeLearning(features, labels)
[m,n] = size(unique(labels));
if n > m
    m = n;
end
if m == 1
    tree = struct('op', '', 'kids', [], 'class', majorityValue(labels));
else
    [best_feature_no, best_threshold, max_gain] = ChooseAttribute(features,labels);
    tree = struct('op', best_feature_no, 'kids', [], 'class', '', 'threshold', best_threshold, 'gain', max_gain);
    best_feature = features(:,best_feature_no);
    minFeature = min(best_feature);
    maxFeature = max(best_feature);
    count = 1;
    for j = 1:2 %Binary Tree
        if j == 1
            features_vi = selectionOnFeatures(minFeature, best_threshold, best_feature, features);
            labels_vi = selectionOnLabels(minFeature, best_threshold, best_feature, labels);
        end
        if j == 2
            features_vi = selectionOnFeatures(best_threshold, maxFeature, best_feature, features);
            labels_vi = selectionOnLabels(best_threshold, maxFeature, best_feature, labels);
        end
        if isempty(features_vi)  %vi是否为空由bound决定
            tree.kids{count} = struct('op', '', 'kids', [], 'class', majorityValue(labels));
        else
            tree.kids{count} = DecisionTreeLearning(features_vi,labels_vi);
        end
        count = count + 1;
    end
end
%% Sub function selectionOnFeatures
function remvFeatures = selectionOnFeatures(lower_bound, upper_bound, best_feature, features)
if lower_bound == min(best_feature)
    features_vi_flag_ = (best_feature(:) >= lower_bound);
    features_vi_flag__ = (best_feature(:) <= upper_bound);
else
    features_vi_flag_ = (best_feature(:) > lower_bound);
    features_vi_flag__ = (best_feature(:) <= upper_bound);
end
features_vi_flag = features_vi_flag_ & features_vi_flag__; 
remvFeatures = features(features_vi_flag==1,:);

%% Sub function selectionOnLabels
function remvLabels = selectionOnLabels(lower_bound, upper_bound, best_feature, labels)
if lower_bound == min(best_feature)
    features_vi_flag_ = (best_feature(:) >= lower_bound);
    features_vi_flag__ = (best_feature(:) <= upper_bound);
else
    features_vi_flag_ = (best_feature(:) > lower_bound);
    features_vi_flag__ = (best_feature(:) <= upper_bound);
end
features_vi_flag = features_vi_flag_ & features_vi_flag__; 
remvLabels = labels(features_vi_flag==1,:);

%% Sub function entropy

function impurity = entropy(labels)
[m,n] = size(labels);
set = unique(labels);
[p,q] = size(set);
impurity_ = 0;
if p == 0 || q == 0
    impurity_ = 0;
else
    if q > p
        p = q;
    end
    if n > m
        m = n;
    end
    for i = 1:p
        count = sum(labels == set(i));
        impurity_ = impurity_ + ((count/m)*log2((count)/m));   %Re mainder(A)
    end
end
impurity = -impurity_;


%% Sub function choose attributes

function [best_feature_no, best_threshold, max_gain] = ChooseAttribute(features, labels)
[m,n] = size(features);
max_gain = 0;
best_feature_no = 0;
for i = 1:n
    [gtemp, tTemp] = gain(features(:,i), labels); %S = labels; Attribute = features(:,i)
    if max_gain < gtemp
        max_gain = gtemp; %For pruning
        best_feature_no = i;
        best_threshold = tTemp;
    end
end

    
%% Sub function gain 

function [g,t] = gain(features, labels)
[m,n] = size(labels);
I = entropy(labels);
maxFeature = max(features);
minFeature = min(features);
%-------------------------------------
%Modification Method 1 (data density & equal division)
sortFeatures = sort(features);
D = 25;
s = zeros(D+1,1);
for i = 1:D+1
    if i == 1
        s(1) = sortFeatures(1);
    end
    if i == D+1
        s(D+1) = sortFeatures(m);
    end
    if i ~= 1 && i ~= D+1
        if floor(m*((i-1)/D)) ~= 0
            s(i) = sortFeatures(floor(m*((i-1)/D)));  % m > D, data density, speed up Algorithm
        else
            s(i) = minFeature + (i-1)*(maxFeature - minFeature)/D;   % m =< D, range equal division
        end
    end
end
T =zeros(D-1,1);
for i = 1:D-1
    T(i) = s(i+1);
end
%-------------------------------------
g = -10000;
for i = 1:D-1  
    sum = 0;
    for j = 1:2 %Binary Tree
        if j == 1
            remvLabels = selectionOnLabels(minFeature,T(i), features, labels);
        end
        if j == 2
            remvLabels = selectionOnLabels(T(i),maxFeature, features, labels);
        end
        [p,q] = size(remvLabels);
        sum = sum +(p/m)*entropy(remvLabels);
    end
    g_ = I -sum;
    if g_ > g
        g = g_;
        t = T(i);
    end
end
    

%%  Sub function majority
function majority = majorityValue(labels)
set = unique(labels);
[m,n] = size(set);
if n > m
    m = n;
end
max = 0;
majority = 0;
for i =1:n
    count = sum(labels == set(i));
    if count > max
        max = count;
        majority = set(i);
    end
end