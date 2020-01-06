# PG_LunarLander_v2  
Simple policy gradient sovling LunarLander_v2 without baseline, wtih temporal structure.  
However, collected batch of trajectories are normalized according to the mean and standard devation.  
This helps the training, performing like using mean baseline.  
The problem was solved at 11840 episode.  
  
What I observed here is that...  
1) simple policy gradient with small learning rate is good enough in the environment  
2) although it takes a lot of episodes to reach a high score  
3) learning speed is expected to be increased by using actor-critic architecture  

# Reward plot
![reward_plot](https://github.com/SHINDONGMYUNG/PG_LunarLander_v2/blob/master/reward_plot.png)  
![reward_plot2](https://github.com/SHINDONGMYUNG/PG_LunarLander_v2/blob/master/reward_plot2.png)
