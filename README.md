# Hybrid Robust Control: DRL + EAND + RLQR 

Algorithm developed in the manuscript entitled “Vision-based robust control framework based on deep reinforcement learning applied to autonomous ground vehicles”, Control Engineering Practice, 2020. 

https://doi.org/10.1016/j.conengprac.2020.104630

# Dependencies
•	Python 

•	Tensorflow 1.4

•	Pygame 1.9.4

•	NumPy 

•	MatplotLib 2.0.2

•	CARLA 0.9.3: please, download CARLA 0.9.3 from https://github.com/carla-simulator/carla/releases/tag/0.9.3 

# Training folder

To train our CNN design: 

1. Copy the folders docs, tensorblock and Training; 

2. Copy CARLA 0.9.3 in Training/sources/carla 

3. Open Training folder and run sh ./scripts/script_TRAIN_CARLA_PPO.sh


To run the trainned CNN: 

1. Open Training folder and run sh ./scripts/load_model.sh

# Test folder 

In the test folder our trained CNN and the hybrid control architecture are provided. To run our model: 

1. Copy the folders docs, tensorblock and Test;

2. Copy CARLA 0.9.3 in Test/sources/carla; 

3. To run the CNN: 

  3.1. Open Test/players_reinforcement/player_PPO_2.py;

  3.2. Comment lines 79 and 80;

4. To run the controllers: 

  4.1. Open Test/players_reinforcement/player_PPO_2.py;

  4.2. Uncomment lines 79 and 80;

  4.3. Select the gain in function lat_control(), line 31;

5. Open Training folder and run sh ./scripts/load_model.sh
