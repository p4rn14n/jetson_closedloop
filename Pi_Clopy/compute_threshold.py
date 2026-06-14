# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:23:09 2025

@author: MurphyLab
"""
#%% COMPUTE THRESHOLD
#%%
import re
import numpy as np
import pandas as pd
import configparser
from pathlib import Path

#%%

def compute_threshold (data_path, section, timestamp_file):
    base_dir = Path(data_path)
    cfg_path = base_dir / 'config.ini'
    # ts_path = base_dir / 'VideoTimestamp.txt'
    # h5_path = base_dir / 'image_stream.hdf5'
    
    # 1) LOAD CONFIG
    cfg = configparser.ConfigParser()
    cfg.read(str(cfg_path))

    sect = section

    # rule = cfg.get(sect, 'roi_operation')   # e.g. "ROI1-ROI2"
    rule = 'TR_R-RFL_R'
    
    
    # split into names and operator
    parts = re.split(r'([-+/%])', rule)       # ['ROI1', '-', 'ROI2']
    roi1, op, roi2 = parts[0], parts[1], parts[2]
    col_combined = f"{roi1}dff{op}{roi2}dff"

    # 2. load the timestamp log
    df = pd.read_csv(base_dir / f'{timestamp_file}.txt', sep='\t') ## Change to VideoTimesampt.txt for normal sessions

    # 3. exclude frames outside trials
    df = df[df['trial'] > 0]

    # 4. compute max ΔF/F per trial
    max_per_trial = df.groupby('trial')[col_combined].max().values

    # 5. find the 75th percentile
    threshold_25 = np.percentile(max_per_trial, 75.0)
    # Create the message
    message = f"Threshold for 25% success ≈ {threshold_25:.4f}"

    # Create and write to a new text file
    with open(base_dir / "threshold_result.txt", "w") as f:
        f.write(message)

    print(f"Threshold for 25% success ≈ {threshold_25:.4f}")
    print("Message saved to threshold_result.txt")