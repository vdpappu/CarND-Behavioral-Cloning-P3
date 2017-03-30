<h2>Behavioral Cloning</h2>

<h3>Required Files</h3>

<h4>Are all required files submitted?</h4>

Submission includes following files:
  * model.py containing the script to create and train the model
  * drive.py for driving the car in autonomous mode
  * model.h5 containing a trained convolution neural network
  * P3_Writeup.md or summarizing the results
  * additional_data.py and dataPreparation.py contains code needed for loading data

<h3>Quality of Code</h3>

<h4>Is the code functional?</h4>

The model provided works with in the simulation with minor modifications in drive.py. Changes to be done are commented in drive.py. model.py also works with minimal changes - modify the paths in the code

<h4>Is the code usable and readable?</h4>

The code is modular and can be reused. Individual modules of the code can be used as-is without major modifications. Comments are included in model.py as needed. 

<h3>Model Architecture and Training Strategy</h3>

<h4>Has an appropriate model architecture been employed for the task?</h4>

A deep neural netwok with stack of convolution and dense layers is used for training. Layers for non-linearity and weight regularization are used for improving the learning process. Data is normalized and resized for daster convergence. layer level and network level hyper parameters are experimented and chosen.

<h4>Has an attempt been made to reduce overfitting of the model?</h4>

Overfitting is handled at two levels:

 * Network level - Dropout layers are added to ensure regularization of weights. This helps in reducing overfitting
 * Data Level - Choice of additional data, simulation steps helps in reducing overfitting. Details of data collection and simulation stratigies are explained in further sections
 
 <h4>Have the model parameters been tuned appropriately?</h4>
 
 Adam optimizes is used, no manual tuning of learning rate
 
 <h4>Is the training data chosen appropriately?</h4>
 
 Data provided has been used for training. Additional data was chosen based on the observations made from the performance of model behavior in the simulator
 
<h3>Architecture and Training Documentation</h3>

<h4>Is the solution design documented?</h4>

An architecture similar to LeNet was chosen as an initial architecture to start with. The performance of ~LeNet with data provided is not at acceptable level - Overfitting(handled by turning network and data as stated above), vehicle not able to steer in the simulator as the data provided has most data with driving_angle close to zero. This is handled by collecting more data. 
A set of experimentation with model architectures has yielded a netwok structure (briefed in next section) that helped running the car in simulator. 

<h4>Is the model architecture documented?</h4>

After experimentations with several model architectures, following architecture is used for generating the submission file:
A stack of convolution layers with ELU non-linearity, dropout and maxpooling and a stack dense layers is used:

Conv(48,5,5), conv(64,3,3) conv(128,3,3)
Each convolution layer is followed by:
dropout(0.5) - for weight regularization to deal with overfitting
elu() - non-linearity
maxpool(2,2) - for spatial learning

dense(512) dense(64) dense(16)
Each dense layer is followed by:
elu() non-linearity

dense(1) - regression head that calculates steering angle

<h4>Is the creation of the training dataset and training process documented?</h4>

 
 
 
 
 
 

