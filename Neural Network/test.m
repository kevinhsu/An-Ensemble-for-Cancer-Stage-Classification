function [confusionmatrix, recall, precision,  accuracy, f_measure] = test(features, targets)
temp_features = featuresPreprocessing(features, 0.25);
temp_targets = targetsPreprocessing(targets, 1, 2);
[new_features, new_targets] = ANNdata(temp_features, temp_targets);
[confusionmatrix, recall, precision, accuracy, f_measure] = cross_validate(10, new_features, new_targets);
