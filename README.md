# 236606project

This is the full code for the course's final project.

final accuracies over test sets:

2X2 - 87% 

4X4 - 29%

5X5 - 7%


evaluate.py - code for accuracy evaluation, following course requirements.

weights_download.py - run to download wieghts for 2X2, 4X4 and 5X5 models.


The final version used for the submission:

transform.py - given images and documents folder, preforms data augmentation, shrades to required tiles number and resizes to same size.

proj_v1.py - run to create model and train over data. outputs graphs for accuracy and loss for train ande test sets.

models.py - contain models for 2X2, 4X4 and 5X5 tiles.
