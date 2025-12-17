import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

scaler = StandardScaler()

PATH = './../../_extracted_features'
for file_name in tqdm(os.listdir(PATH)):
  full_path = f'{PATH}/{file_name}'
  
  raw_features = np.load(full_path)
  scaler.partial_fit(raw_features[:, 4:])
  
joblib.dump(scaler, './outputs/feat_scaler.joblib')
  
# for file_name in tqdm(os.listdir(PATH)):
#   full_path = f'{PATH}/{file_name}'
  
#   raw_features = np.load(full_path)
#   np.save(full_path, np.hstack((raw_features[:, :4], scaler.transform(raw_features[:, 4:]))))
