
# coding: utf-8

# In[13]:


import pandas as pd
import os

def fix_path_names():
    """Run when in /Code folder. Fixes only file if all paths are messed up. """
    path = os.getcwd().split(os.sep)[:-1] + ["driving_log.csv"]
    path_str = (os.sep).join(path) 
    data = pd.read_csv(path_str, header=None) 
    front = [x.split(os.sep)[-1] for x in data[0]]
    left = [x.split(os.sep)[-1] for x in data[1]]
    right = [x.split(os.sep)[-1] for x in data[2]]
    data[0] = front
    data[1] = left
    data[2] = right
    data.to_csv(path_str, index=False, header=False)
fix_path_names()

	
