# Model Training and Evaluation

## Initial Experimentation
In the [first notebook](./01_scratch_model.ipynb), I experimented with training a Convolutional Neural Network (CNN) from scratch designed using the Sequential API in Keras. 

In the [second notebook](./02_transfer_learning_model.ipynb), I experimented with transfer learning. I used a a pre-trained neural network (`MobileNet`), froze the convolutional layers and added my own dense layers. 

After several attempts to tweak the transfer learning model, the accuracy scores for the validation set never got better than about 0.35 so I decided the final model should be built from scratch.

## Final model training approach
In the [third notebook](./03_final_model.ipynb), I built the final model and experiment with the following:

- Using the `class_weight` parameter during training to adjust the loss based on the class imbalance (not a useful addition)
- Data Augmentation
- Modifying the Learning Rate
- Modifying the Dense Layers: my original model was built with 3 layers in total (1 output layer and 2 layers with 128 and 64 units respectively).
- Introducing regularization via dropout
- Changing the number of epochs

In the end I trained and evaluated 3 CNN models. All the models had a similar architecture and the same learning rate. See the [third notebook](./03_final_model.ipynb) for details.


## Results

| Model                            | Final score     | 
| -------------------------------- | --------------- |
| No Dropout, No Data Augmentation | 0.5461          |
| Dropout, No Data Augmentation    | 0.5704          | 
| No Dropout, Data Augmentation    | 0.5713          | 

## Evaluation and next steps
Our final scores are disappointing and a little confusing. This is the second time I have run the notebook and have come to slightly different conclusions. 

I was unable to handle class imbalance effectively: I would like to try some of the other techniques I found during EDA to see if these produce better results.

I would like to spend more time experimenting with different model architectures.

I would like to try this task with a different dataset: the images are very small, the quality is questionable and some of the categorisation is dubious. 