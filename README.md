# Neural Network Classification of actions performed by humans in short video clips

## Table of Contents
- [General](#general)
- [Dataset](#dataset)
- [Pre-processing](#pre-processing)
  - [Split in training, validation, test set](#split-in-training-validation-test-set)
  - [Silhouette extraction](#silhouette-extraction)
- [Data Augmentation](#data-augmentation)
  - [Scaling](#scaling)
  - [Binarisation](#binarisation)
- [Architecture](#architecture)
  - [Final Architecture:](#final-architecture)
- [Loss](#loss)
- [Training](#training)
- [Classification & Testing](#classification--testing)
- [Generalisation](#generalisation)
- [Further Steps](#further-steps)


## General
A neural network was to be designed to classify movement of humans recorded as short video clips, specifically a simple [Weizmann dataset](https://mega.nz/file/omlVQJrB#OJGx0r90H4ymNt7ZhikVGmroQFN0TDvgnH5ssIjRuJ0) was used. 

The finally used and trained model can be found [here](https://github.com/RaphaelNekam/Deep_Learning_AE/blob/main/models/CNNGRU_21_epochs_patience_5.pth). The full Google Colab Python Notebook can be found [here](https://github.com/RaphaelNekam/Deep_Learning_AE/blob/main/NN_Movement_Classification.ipynb).

## Dataset
The dataset consists of 93 videos of different people performing 10 categories of movements. Each video is at least 36 frames long with a resolution of 180x140 pixels. Here are some example videos:

| Name | Movement | Video |
| :--- | :--- | :--- |
| **Eli** | Jumping Jack | ![eli_jack](https://github.com/user-attachments/assets/4f15089a-0149-4887-8a75-45f9261b43b4) |
| **Ira** | Walk | ![ira_walk](https://github.com/user-attachments/assets/6e281c67-a21c-4040-9223-6c57aa17aa5a) |
| **Iyova** | Wave | ![lyova_wave1](https://github.com/user-attachments/assets/926c15f9-c22c-440b-8b61-cac7d728073a) |
| **Shahar** | Bend | ![shahar_bend](https://github.com/user-attachments/assets/b412abf4-ce84-49c3-874c-00e10d36a9a7) |

## Pre-processing
### Split in training, validation, test set

The dataset is divided into training, validation, and test sets. The training set is used to adjust the network's weights, while the validation set evaluates the impact of these changes on unseen data. The test set is exclusively used for the final accuracy assessment and is never seen during training.

To achieve this split, one video from each group was randomly selected for the test and validation sets, with the remaining videos used for training. This resulted in:

- Training set: 73 videos, 7-8 per movement
- Validation set: 10 videos, 1 per movement
- Test set: 10 videos, 1 per movement

### Silhouette extraction
Videos are provided in RGB format, but colors were deemed unnecessary. Instead, relevant information was extracted before feeding into the network. Two approaches were considered:

1. Foreground Extraction: Pixels of the foreground are set to 1, and background pixels to 0, resulting in silhouettes of the person.

2. Moving Pixels Extraction: Pixels that change between frames are set to 1, and static pixels to 0, creating "movement silhouettes."

Using binary images reduces the number of channels, improving training time. Both methods pre-filter important video information. Approach (2) performed better in initial tests and was used going forward, enhancing invariance to lighting, colors, background, and clothing.


Picture of a moving silhouette for a running person:
<img width="145" height="152" alt="image" src="https://github.com/user-attachments/assets/ffcda3eb-abf6-4680-8e74-25087226f26a" />



## Data Augmentation

Due to the small sample size, random transformations were applied to increase the dataset:

- Random Frame Selection: 36 consecutive frames are taken from each video at a random start point.
- Random Cropping: Frames are cropped from 180x144 to 128x128 pixels.
- Vertical Flipping: Videos are flipped vertically with a 50% chance.
- Random Rotation: Videos are rotated between -20 and +20 degrees.

$$
(100-36) * (180-128) * (144-128) * 2 * 40 = 4,259,840
$$

These transformations can create up to 4,259,840 variations for a 100-frame video, adding invariance to rotation and subject location.


Many of these videos might be quite similar, as changes in 1 px, 1° or 1 frame are minor. However, it still allows some form of data augmentation, with the exact same images unlikely to be used again as 10,000 samples from 73 videos of 4M different variations were drawn.

### Scaling
After cropping videos from 180x144 to 128x128 pixels, they were further scaled down to 64x64 pixels, as it was found to improve training times, while still carrying all important information.


### Binarisation
After the silhouette extraction, pixels are either 255 or 0. By utilizing the rotation transformation used, some pixels were numbers in between those. For better training and more cohesive input data, all pixels are normalized to either 0 or 1, depending on a set threshold.

---

## Architecture
The final architecture consists of convolutional layers, followed by a GRU layer, again followed by fully connected layers. This proved to be a well-working setup.

The initial neural network consisted solely of RNN layers followed by fully connected layers. RNNs were chosen for their "memory" capabilities, but they did not achieve high accuracy, likely due to the loss of spatial information when feeding input data. To address this, a convolutional layer was added before the RNN, with additional layers added later, significantly improving performance. The RNN was then replaced with a GRU, which better mitigates the vanishing gradient problem. Although LSTM was tested, it did not offer significant improvements over GRU, so GRU was retained for its simpler structure, keeping the model efficient.


### Final Architecture:

1. Convolutional Layers

    Used to extract spatial information from the input frames. This is helpful to identify where the person is in the image and how it is moving. ReLU activation functions were used as they are a common choice that proofed to be suitable during testing.

  - Conv1
  - Conv2
  - Conv3

2. Max Pooling

    Max Pooling is used to scale down the dimensionalty.

3. GRU Layer

    The GRU layer is used to make sense of consecutive frames. It allows to "connect" the spatial features of different frames together, identifying motions. It "makes sense" of consecutive information and can "memorise" what happened in frames before. GRU was preferred over a "normal" RNN as it mitigates the vanishing gradient problem. A 2-time stacked GRU was used as it showed to improve performance, allowing to extract more detailed information.

4. Fully Connected Layers

    These layers make sense of the information extracted by the GRU layer, allowing to identify probabilities for the 10 movement classes possible. They translate the GRU output into th emost likely movement class. Adding a third FC layer, allowed to push the accuracy from about 95% to 99%. Adding further fully connected layers might further improve this. ReLU activation functions were used for similar reasons as the ones for the Convolutional Layers.

  - FC1
  - FC2
  - FC3


---


## Loss

Cross-entropy loss was used for training the model. This is a common method used for multi-class classification problems, which this example is (deciding between 10 different classes of motions). The loss function proved to work during the entirety of building this NN and was hence kept.


---



## Training

The model is trained on the training set, a set of 73 videos which are augemnted using a data loader and the transformations described above. The final model was trained for 21 epochs (as stopped by the scheduler), on 10000 samples with a batch size of 32 (the batch size used is a common good practice).

The step size starts at 0.0001, decreasing as the scheduler moves forward. Other step sizes were tried (1.000 - 0.000001), but none have proven to be working better than 0.0001. To compare them, convergence graphs were analysed next to each other. The chosen 0.0001, together with a scheduler, showed to be a sweet spot.

A scheduler was used to half the step size every 10 epochs, as this proved to be a quite good approach. Training the model without a scheduler was tried first, but this led to worse results. The scheduler was changed to different numbers of epochs (30 and 5) but 10 proved to be a decent sweet spot for convergence.

For each epoch, the loss for the validation data is calculated as well. Using "early stopping" the training is stopped if the validation loss does not improve for 5 epochs, which should lower the risk of overfitting. Stopping the model earlier showed to indeed offer better results for the validation data classification.

The images below show how the loss for both training and validation data changed over the training epochs. It shows the model converging as training proceeds. These graphs are for the final model provided.

<img width="850" height="440" alt="image" src="https://github.com/user-attachments/assets/7ca1c1c1-c856-43ac-be8e-8ab60c650607" />


## Classification & Testing

To verify the model to work well, a test dataset of 10 videos, yet unseen by the model, were augmented using the same transformations applied to the training set. It was also tested to apply only the silhouette extraction, cropping & frame extraction (as this data does not need to be augmented), however this lead to worse classifiation results for the validation set. Hence, both validation and test data was transformed in the exact same way as the training data was.

The model was then to classify various samples to estimate its accuracy in classifying the videos correctly. Further, a confusion matrix was created, to determine which motions were commonly confused with one another.

The general performance was found to be ~99.5% accuracy. The confusion matrix shows that mostly videos of motion 9 (skipping) were confused to be videos of class 1 (running) and 2 (jumping). This makes sense, as skipping is somewhat a combination of these two motions.

<img width="284" height="211" alt="image" src="https://github.com/user-attachments/assets/9b8c835c-a765-41a4-8af6-b69b565d2c40" />


The graph below shows one transformation of each of the 10 test videos, together with the true and predicted label.

<img width="211" height="705" alt="image" src="https://github.com/user-attachments/assets/fa184aad-68be-468a-958f-41458ef2e7a6" />


---

## Generalisation
I do not expect this model to generalise very well. I believe due to the limited dataset this is almost impossible. Applying the transformations as described, allows to somewhat augment the data, however, starting from only 73 videos, limits what is achievable. The model has good performance on similar data (same camera, similar angle, etc.). But I do not think that other videos of other recordings will have the same results. It might still perform okay, but I do not expect to to reach an accuracy of correctly classifying videos with a 99% accuracy as it does for the test data chosen in this example.

---

## Further Steps

While the model already works well for classifying motions from the sample videos, further improvements could be made.

1. The videos could be adjusted to different speeds, skipping or adding frames. This could make the model more robust for motions of varying speeds.

2. The videos are currently only rotated in ranges of +/-20°. This range could further be extended. However, it is likely that those are the actual angles of videos, as cameras that are more tilted are unlikely.

3. At the moment, all videos are cropped from 180x144 to 128x128 pixels. Further, all videos seem to be filmed from a similar distance to the subject. To add more "distance invariance", the videos could be "zoomed in" or out improve this behaviour.

4. Adding noise. Sometimes, adding noise to videos help to train the model more sucessfully, as it learns to "only" focus on important information. This could also be tested for this model

5. Adding random obstructions. Pixel sections could be removed from the videos to simulate objects obstructing the view. This could e.g. simulate cars, pillars etc. This might add to the robustness of the model

6. Adding more layers could improve the model further. However, it should also be kept as simple as possible.

If I had more time and a more powerful device to run my training on (such as the A100), I would probably add more layers, further trying more implementations such as varying activation functions. The confusion matrix also shows, that most confusions happen at class 9 (skipping), which is commonly identified as class 1 (running) or 2 (jumping). This confusion does make sense, as skipping as a motion that combines both running and jumping, but would also be an interesting topic to look into. Why does this misclassification happen? We specifically for skipping? In doing so, the performance of the model could be further improved.

