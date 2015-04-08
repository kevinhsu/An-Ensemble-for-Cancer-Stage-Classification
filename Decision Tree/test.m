function [confusionmatrix, recall, precision,  accuracy, f_measure] = test(features, targets)

[new_features, new_targets] = dataPreprocessing(features, 0.25, targets,1, 2);

[confusionmatrix, recall, precision, accuracy, f_measure] = cross_validate(10, new_features, new_targets);
