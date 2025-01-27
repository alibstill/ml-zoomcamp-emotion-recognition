# EDA

In this section I carry out some exploratory data analysis, do some preprocessing given the results of this analysis and think about how I might solve the class imbalance problem.

1. **Stage 1**: Exploration

In the [first notebook](01_exploratory_data_analysis.ipynb) I 
- download the data
- inspect random images visually 
- look at class distribution
- explore image meta data e.g. format, size, resolution
- analyse quality by looking at a measure of the noise of the image
- identify duplicate images

The dataset is quite large: there are 28,709 training images and 7,178 test images. The training images are all grayscale, 48x48 JPGs, which is a small image size: most pre-trained models expect much larger sizes, for example Xception works with images that are 299x299 pixels.

The training dataset is also very imbalanced. Most of the images are categorised as "happy": 
- happy: 0.251315
- neutral: 0.172942
- sad:0.168240
- fear:0.142708
- angry: 0.139155
- surprise 0.110453
- disgust: 0.015187

Using a measure of noise, I was able to identify very dark very high contrast images for removal.

The package [imagehash](https://github.com/JohannesBuchner/imagehash) helped me to identify duplicate images.

2. **Preprocessing**

In the [second notebook](./02_preprocessing.ipynb), the duplicate and low quality images are removed from the dataset and a clean dataset is saved. This cleaned dataset is the input for all the models.

This reduced the total training set from 28,709 -> 26,767.

I also extract the test and validation set.

The dataset was already divided into test and train but I wanted to split up the test data further into a validation set and a test set. It seems that this was the original intention: the test files were already labelled with a suffix of either `PrivateTest` and `PublicTest`. I split the test set on these suffixes and got an equal number of files.

I applied the same preprocessing to the validation and test set.

This reduced the validation set from 3,589 -> 3,478.

It reduced the test set from 3,589 -> 3,475.


3. **Class Imbalance**

In the [third notebook](./03_class_imbalance.ipynb), I explored some means of addressing class imbalance, including:

- data augmentation
- manipulating batch composition
- adjusting the loss function

I opted to experiment with adjusting the loss function first, because this seemed the most straightforward.