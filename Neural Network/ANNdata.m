function [data, labels] = ANNdata(x, y)
x = scaled(x, 0, 1);
y = scaled(y, 0, 1);
z = [x, y];
[m,n] = size(x);
shuffledArray = z(randperm(size(z,1)),:);
data = shuffledArray(:,1:n);
labels = shuffledArray(:,n+1);
labels = getLabels(labels);
data = data';
labels = labels';

%% --
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

function scaled_data = scaled(data, minv, maxv)   %如果只训练一次nn然后用来检测test数据，需要注意新的test是否超过max或者min
scaled_data = data - min(data(:));
scaled_data = (scaled_data/range(scaled_data(:)))*(maxv-minv);
scaled_data = scaled_data + minv;
