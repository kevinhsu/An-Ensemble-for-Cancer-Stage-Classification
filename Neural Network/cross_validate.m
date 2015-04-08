function [confusionmatrix, recall, precision,  accuracy, f_measure] = cross_validate(num_of_buckets, data, labels)
%num_of_buckets = 10; %default
%NET  = newff(data, labels, 25);

NET  = newff(data, labels, [50, 50],{'tansig', 'purelin', 'logsig'});

set = unique(labels);
[p,q] = size(set);
if p <= q
    p = q;
end
uniq = p; 
confusionmatrix = zeros(p, p);

[m,n] = size(data);
bucket =  floor(n/num_of_buckets);

recall = 0;
precision = 0;
folds = [];

max_ = bucket;
min_ = 0;
leftover = 0; 
for i = 1:num_of_buckets
    structure = struct('network', [], 'predicted', [], 'truth', []);
    
    test_data = data(:,min_+1:max_);
    train_data = selectRelevant(data, min_+1, max_);
    test_labels = labels(:,min_+1:max_);
    train_labels = selectRelevant(labels, min_+1, max_);
    
    [network, TR] = train(NET, train_data, train_labels);
    structure.network = network;
  
    structure.predicted = sim(network, test_data);
    structure.truth = test_labels;
    
    fold(i) = structure;
    
    min_ = floor(min_ + bucket);
    if (i ~= 10) 
        if(i ~= 9)
            max_ = floor(max_ + bucket);
        else
            max_ = floor(max_ + bucket);
            leftover = (q - max_);
            max_ = max_ + leftover;
        end
    end
end


for i = 1: num_of_buckets   
    if (i ~= 10)  
        for j = 1:bucket
            class = find(fold(i).predicted(:,j) == max(fold(i).predicted(:,j)));
            actual =  find(fold(i).truth(:,j) == max(fold(i).truth(:,j)));
            confusionmatrix(actual, class) =  confusionmatrix(actual, class) + 1;
        end
    else 
        last_batch = bucket+ leftover;
        for j = 1:last_batch
            class = find(fold(i).predicted(:,j) == max(fold(i).predicted(:,j)));
            actual =  find(fold(i).truth(:,j) == max(fold(i).truth(:,j)));
            confusionmatrix(actual, class) =  confusionmatrix(actual, class) + 1;
        end
    end
end
for h = 1:p
    tp = 0;
    fp = 0;
    for j = 1:p
        if (h == j)
            tp = tp + confusionmatrix(j, h) ;
        else
            fp = fp + confusionmatrix(j, h);
            
        end
    end
    precision = precision + (tp / (tp + fp)) ;
end
precision = precision / uniq;

for h = 1:p
    tp = 0;
    fn = 0;
    for j = 1:p
        if (h == j)
            tp = tp + confusionmatrix(h,j) ;
        else
            fn = fn + confusionmatrix(h,j);
        end
    end
    recall = recall + (tp / (tp + fn)) ;
end
recall = recall / uniq;
accuracy = sum(diag(confusionmatrix))/sum(sum(confusionmatrix));
f_measure = 2 * ((precision * recall) / (precision + recall));
%%
function relevant = selectRelevant(data, min, max)
[m,n] = size(data);
relevant = zeros(0,0);
for i = 1:min-1
    relevant(:,end+1) = data(:,i);
end

for i = max+1:n
    relevant(:,end+1) = data(:,i);
end







