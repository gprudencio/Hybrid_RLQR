# Hybrid Robust Control: DRL + EAND + RLQR 

Algorithm developed in the manuscript entitled “Vision-based robust control framework based on deep reinforcement learning applied to autonomous ground vehicles”, Control Engineering Practice, 2020. 

https://doi.org/10.1016/j.conengprac.2020.104630

Please, download CARLA 0.9.3 from https://github.com/carla-simulator/carla/releases/tag/0.9.3, and place the folders in Experimental/learning/sources/carla folder. 

# Dependencies
•	Python 
•	Tensorflow 1.4
•	Pygame 1.9.4
•	NumPy 
•	MatplotLib 2.0.2
•	CARLA 0.9.3

# Training folder

For training the CNN, please open learning folder and run sh ./scripts/script_TRAIN_CARLA_PPO.sh 

# Test folder 

The trained CNN is presented for testing. Please, open the learning folder and run sh ./scripts/model.sh
For controlling the car 
