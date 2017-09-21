A Team - System Integration Project 
===================================

![A Team](imgs/ateamlogo.png "There is no Plan B.")

Udacity Term 3 Final Capstone Project

## Traffic Light Classifier Documentation

The classifier part consists of 3 files:

* *classifier_trainer.py*: contains class **TLClassifier_Trainer**. This part is the core of the training stage. It reads in sample files and - using a **LeNet** architecture - does the trainig stage and finally stores the model in respective files (model.* in *light_classification* directory). For now, labeling the images is done by hand, just putting the images in respective directories. The directory structure is as follows:
	- top dir is *pics*
	- the images are stored in directories named GREEN, YELLOW, RED and UNKNOWN, respectively:

	(...) light_classification
								
					--> pics
								--> GREEN
								--> YELLOW
								--> RED 
								--> UNKNOWN

The classifier reads in the images from each directory; by knowing the directory the labels are derived.

* *tl_classifier.py*: (Provided by Udacity): contains TLClassifier() class. This  is the *"Client"* class; it loads the model stored in the training step and classifies incoming images. It reuses the logits from the trainer (Not strictly required, but avoids code duplication)

* *train.py*: short helper class. Use this is you need to invoke training. It just instantiates the training class and invokes training.  