function [predicted_targets, inconsistent_index] = predictedTargets(features, targets, tree)

[m, n] = size(features);
predicted_targets = zeros(1,m);
predicted_gain = zeros(1,m);
inconsistent_targets = zeros(1,m);
for j = 1:m
    [predicted_targets(j), predicted_gain(j)]= classify(features(j,:), tree);
    if targets(j) ~= predicted_targets(j)
        inconsistent_targets(j) = 1;
    end
end

inconsistent_index = find(inconsistent_targets == 1);




%%
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