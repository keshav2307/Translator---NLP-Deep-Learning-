# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:55:20 2020

@author: Keshav
"""

import os

# loading the data from files to dataset

def load(path):
    
    # load dataset
    
    input_file = os.path.join(path)
    with open(input_file, "r") as file:
        data = file.read()

    return data.split('\n')