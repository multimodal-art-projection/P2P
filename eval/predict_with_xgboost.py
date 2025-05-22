import os
import json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import yaml
import argparse

def load_common_yaml(yaml_path="common.yaml"):
    """加载common.yaml文件，获取评估项列表"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        common_yaml = yaml.safe_load(f)
    return [item['description'] for item in common_yaml['checklist']]

def predict_score(feature_vector, model):
    """使用模型预测分数"""
    return model.predict([feature_vector])[0]
