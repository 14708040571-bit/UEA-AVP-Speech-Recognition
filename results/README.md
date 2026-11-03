
# Experimental Results

This folder contains the experimental results of my speech recognition system.

## Validation Results
- Selected-model validation accuracy: 81.25%
- Selected feature: 13 MFCC coefficients
- Selected epoch: 16
- Validation loss at the selected epoch: 0.6603

The validation experiment selected epoch 16 because it achieved the lowest validation loss (0.6603), with a validation accuracy of 81.25%. Although epoch 17 reached 82.50% validation accuracy, its validation loss was slightly higher. The final CNN was then trained separately on all 400 development recordings for 16 epochs.

## Independent Test Results
- Test recordings: 100
- Correct predictions: 33
- Incorrect predictions: 67
- Test accuracy: 33.00%
- Test loss: 3.5428

## Figures
- `training_validation_accuracy.png`: Training and validation accuracy
- `training_validation_loss.png`: Training and validation loss
- `validation_confusion_matrix.png`: Validation confusion matrix
- `final_test_confusion_matrix.png`: Independent test confusion matrix

## Discussion
The selected MFCC-based CNN achieved 81.25% validation accuracy but only 33.00% accuracy on the independent test set.

The large difference suggests limited generalisation to new recordings and possible overfitting to the development data.
