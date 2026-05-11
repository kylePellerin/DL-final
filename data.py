import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os 
import sys
import glob
from PIL import Image 
import numpy as np
import torchvision.transforms as T

countries_mapping = {
    "N/A": 0,
    "Argentina": 1,
    "Australia": 2,
    "Austria": 3,
    "Budapest": 4,
    "Chile": 5,
    "China": 6,
    "Colombia": 7,
    "Czech Republic": 8,
    "Denmark": 9,
    "Ecuador": 10,
    "Egypt": 11,
    "France": 12,
    "Germany": 13,
    "Ghana": 14,
    "Greece": 15,
    "Hong Kong": 16,
    "Hungary": 17,
    "Iceland": 18,
    "India": 19,
    "Ireland": 20,
    "Italy": 21,
    "Japan": 22,
    "Jordan": 23,
    "Kazakhstan": 24,
    "Kenya": 25,
    "Madagascar": 26,
    "Mexico": 27,
    "Mongolia": 28,
    "Morocco": 29,
    "Nepal": 30,
    "Netherlands": 31,
    "New Zealand": 32,
    "Panama": 33,
    "Peru": 34,
    "Portugal": 35,
    "South Africa": 36,
    "South Korea": 37,
    "Spain": 38,
    "Sweden": 39,
    "Switzerland": 40,
    "Taiwan": 41,
    "Tanzania": 42,
    "Trinidad & Tobago": 43,
    "Turks & Caicos Islands": 44,
    "Turks and Caicos Islands": 45,
    "United Kingdom": 46,
    "United States": 47
}

majors_mapping = {
    "N/A": 0,
    "Africana Studies": 1,
    "Anthropology": 2,
    "Art History": 3,
    "Asian Studies": 4,
    "Biochemistry": 5,
    "Biology": 6,
    "Chemistry": 7,
    "Chinese": 8,
    "Classics": 9,
    "Computer Science": 10,
    "Digital and Computational Studies": 11,
    "Earth and Oceanographic Science": 12,
    "Economics": 13,
    "Education": 14,
    "English": 15,
    "Environmental Studies": 16,
    "Francophone Studies": 17,
    "French": 18,
    "Gender, Sexuality, and Women's Studies": 19,
    "German": 20,
    "Government and Legal Studies": 21,
    "Hispanic Studies": 22,
    "History": 23,
    "Italian Studies": 24,
    "Japanese": 25,
    "LCL Studies": 26,
    "Latin American Studies": 27,
    "Mathematics": 28,
    "Middle Eastern and North African Studies": 29,
    "Music": 30,
    "Neuroscience": 31,
    "Philosophy": 32,
    "Physics": 33,
    "Psychology": 34,
    "REE Studies": 35,
    "Religion": 36,
    "Romance Language and Literatures": 37,
    "Russian": 38,
    "Sociology": 39,
    "Theater and Dance": 40,
    "Visual Arts": 41
}

class_mapping = {
    "2029": 0,
    "2028": 1,
    "2027": 2,
    "2026": 3,
}


"""Paths"""
image_dir = "./data/images/*"
csv_path = "./data/data_info.csv"

def load_data(image_dir, csv_path):
    image_paths = glob.glob(image_dir)
    data_info = pd.read_csv(csv_path)

    data_dict = {} 

    # Iterate through the CSV and build the data dictionary
    for index, row in data_info.iterrows():
        image_id = row["Image ID"]
        class_label = row["ClassVals"]
        major_labels = row["MajorVals"]
        country_labels = row["CountryVals"]
        # print(f"Image ID: {image_id}, Class Label: {class_label}, Major Labels: {major_labels}, Country Labels: {country_labels}")

        if "," in major_labels:
            major_labels = major_labels.split(", ")
            major_labels = [int(label) for label in major_labels]
        else:        
            major_labels = [int(major_labels)]

        image_path = f"./data/images/{image_id}"

        data_dict[image_path] = {
            "class_label": int(class_label),
            "major_labels": major_labels,
            "country_labels": int(country_labels),
        }
        
        if f"./data/images/{image_id}" not in image_paths:
            print(f"Image ID: {image_id} not found in image paths")

    return data_dict

def process_data(data_dict, device, transforms):
    for key, value in data_dict.items():
        img = Image.open(key).convert('RGB')          
        img_tensor = transforms(img).to(device)   

        zeros_class = torch.zeros(4, dtype=torch.float32).to(device)
        zeros_country = torch.zeros(48, dtype=torch.float32).to(device)
        zeros_major = torch.zeros(42, dtype=torch.float32).to(device)

        class_label_one_hot = zeros_class.clone()
        class_label_one_hot[value['class_label']] = 1.0
        country_label_one_hot = zeros_country.clone()
        country_label_one_hot[value['country_labels']] = 1.0   
        major_label_one_hot = zeros_major.clone() #this shoudl have been outside loop
        for major_label in value['major_labels']:
            major_label_one_hot[major_label] = 1.0        

        # store in dict 
        data_dict[key]['image_tensor'] = img_tensor
        data_dict[key]['class_label_one_hot'] = class_label_one_hot
        data_dict[key]['country_label_one_hot'] = country_label_one_hot
        data_dict[key]['major_label_one_hot'] = major_label_one_hot

    return data_dict
    
class BowdoinData(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.image_paths = list(data_dict.keys())
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        data = self.data_dict[image_path]
        return data['image_tensor'], data['class_label_one_hot'], data['country_label_one_hot'], data['major_label_one_hot']

"""
Statistics for the Images
Mean: tensor([0.5173, 0.4501, 0.4103])
Std: tensor([0.2840, 0.2643, 0.2671])
"""
