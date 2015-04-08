function [confusionmatrix, recall, precision,  accuracy, f_measure] = cross_validate(num_of_folds, features, targets)

set = unique(targets);
[p,q] = size(set);
if p <= q
    p = q;
end
uniq = p; 
confusionmatrix = zeros(p, p);

[m,n] = size(features);
bucket =  floor(m/num_of_folds);

alllabels = getLabels(targets);
[p,q] = size(alllabels);

recall = 0;
precision = 0;
folds = [];

max = bucket;
min = 0;
 
for i = 1:num_of_folds    
    for h = 1:q
        structure = struct('tree', [], 'predicted', zeros(1, max - min), 'gain', zeros(1, max - min) , 'truth', []);
	    % 1-vs-N multiple classification
        labels_h = alllabels(:,h);
        test_features = features(min+1:max,:);
        train_features = selectTraining(features, min+1, max);
        test_labels = labels_h(min+1:max, :);
        train_labels = selectLabels(labels_h, min+1, max);
        
        tree = DecisionTreeLearning(train_features, train_labels);
        
        structure.tree = tree;
        for j = 1:(max - min)
             [structure.predicted(j), structure.gain(j)] = classify(test_features(j,:), tree);
        end
        structure.truth = test_labels;
        list_of_structures(h) = structure;
     end
    folds{i} = list_of_structures;
    min = floor(min + bucket);
    max = floor(max + bucket);
    if max + bucket > m  %防止不能被整数
        max = m;
    end
end

max = bucket;
min = 0;

for i = 1: num_of_folds   
    for j = 1:(max - min)
        prediction = getClass(folds{i}, j); %getClass
        actual = targets(j + min);
        confusionmatrix(actual, prediction) = confusionmatrix(actual, prediction) + 1;
    end
    min = floor(min + bucket);
    max = floor(max + bucket);
    if max + bucket > m  %防止不能被整数
        max = m;
    end
end
 
for h = 1:q
    tp = 0;
    fp = 0;
    for j = 1:q
        if (h == j)
            tp = tp + confusionmatrix(j, h) ;
        else
            fp = fp + confusionmatrix(j, h);
        end
    end
    precision = precision + (tp / (tp + fp)) ;
end
precision = precision / uniq;

for h = 1:q
    tp = 0;
    fn = 0;
    for j = 1:q
        if (h == j)
            tp = tp + confusionmatrix(h,j) ;
        else
            fn = fn + confusionmatrix(h,j);
        end
    end
    recall = recall + (tp / (tp + fn)) ;
end

recall = recall / uniq;
accuracy = sum(diag(confusionmatrix)) / sum(sum(confusionmatrix));
f_measure = 2 * ((precision * recall) / (precision + recall));
 
%% labels_m = getLabels(labels)
function labels_m = getLabels(labels)
set = unique(labels);
[p,q] = size(set);
[m,n] = size(labels);
labels_m = zeros(m, p);
for i = 1:p
    label_i = zeros(m,n);
    current_label = set(i);
    for j = 1:m
        if (current_label == labels(j))
            label_i(j) = 1;
        end
    end
    labels_m(:,i) = label_i;
end

%% function trainingdata = selectTraining(data, test)
function trainingdata = selectTraining(data, min, max)
[m,n] = size(data);
trainingdata = [];

for i = 1:min-1
    trainingdata(end+1,:) = data(i, :);
end

for i = max+1:m
    trainingdata(end+1,:) = data(i, :);
end


%% function traininglabels = selectLabels(labels, min, max)
function traininglabels = selectLabels(labels, min, max)
[m,n] = size(labels);
traininglabels = [];

for i = 1:min-1
    traininglabels(end+1,:) = labels(i,:);
end

for i = max+1:m
    traininglabels(end+1,:) = labels(i,:);
end

%% function [class, gain]= classify(instance, decisionTree)
function [class, gain]= classify(instance, decisionTree)
node = decisionTree; 
while (isempty(node.kids) == 0)
    op = node.op;
    value = instance(op);
    oldnode = node; 
    if value <= node.threshold
        node = node.kids{1};
    else
        node = node.kids{2};
    end
end
class = node.class;
gain = oldnode.gain;

%% function class = getClass(fold, index)
function class = getClass(fold, index)
[m, n] = size(fold);
predicted = zeros(1, n);
onegains = [];
for i = 1:n
    structure = fold(i);
    predicted(i) = structure.predicted(index);
    if (predicted(i) == 1)
        onegains(i) =  structure.gain(index);
    end
    gains(i) =  structure.gain(index);
end

[m] = size(onegains); 
% 若class结果不一致，则根据上一层的gain大小来选取较为合适的class
if (m ~= 0 )
    max_ = max(onegains);
    class = find(onegains == max_);
else 
    max_ = max(gains);
    class = find(gains == max_);
    [m, n] = size(class); 
    if m > n || n > m 
        class = class(randi(numel(class)));
    end 
end 