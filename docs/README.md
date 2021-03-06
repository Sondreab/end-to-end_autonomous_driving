# TDT4265_final_project
Final term project in the course TDT4265 - Computer vision and deep learning of spring 2019 at NTNU

Implemented by Sondre Aleksander Bergum, Martin Madsen and Filip Schjerven.

## Instructions
Train a model by running model.py. By default the code saves it as "model.h5".

To change what training data your model trains on you must manually change to the correct .csv and image folder in main, the default is to train a model on data from both tracks.

We construct visualizations of layer activations from a random training-image after training a model that are saved in docs/plots.

Run models by executing "python3 drive.py <modelname>" in terminal. Some good models are provided for you already. 

## Resources
[Project description](https://www.overleaf.com/read/xgqfysbtbcpd) (Project 3)  
[Provided code](https://drive.google.com/file/d/1hKVc4METKj2aQy4yC3xnP8Dwc4zEd-Cn/view)  

Litterature:  
https://arxiv.org/pdf/1704.07911.pdf <Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car>  
https://arxiv.org/pdf/1604.07316.pdf <End to End Learning for Self-Driving Cars>  
https://arxiv.org/pdf/1710.03804.pdf <End-to-End Deep Learning for Steering Autonomous
Vehicles Considering Temporal Dependencies>  
https://selfdrivingcars.mit.edu/ <MIT 6.S094: Deep Learning for Self-Driving Cars>  
http://cs231n.stanford.edu/reports/2017/pdfs/626.pdf <Self-Driving Car Steering Angle Prediction Based on Image Recognition>  
https://arxiv.org/pdf/1608.01230.pdf + https://github.com/commaai/research <Learning a Driving Simulator>  
https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/ <Explaining How End-to-End Deep Learning Steers a Self-Driving Car>  
https://devblogs.nvidia.com/deep-learning-self-driving-cars/ <End-to-End Deep Learning for Self-Driving Cars>  


Other resources:  
https://blog.coast.ai/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a  
https://github.com/tech-rules/DAVE2-Keras  
https://github.com/pszczesnowicz/SDC-P3-Behavioral-Cloning/blob/master/model.py

## License
GPLv2
