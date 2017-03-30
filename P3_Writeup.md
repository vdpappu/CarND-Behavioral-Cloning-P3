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
