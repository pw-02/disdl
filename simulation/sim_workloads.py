#based on batch size of 128 and imagenet dataset

#imagenet dataset
#batch size of 128
import os
import csv
from typing import Dict, List
import numpy as np
import time

def calculate_elasticache_serverless_cost(
    average_gb_usage: float,
    duration_hours: float = 1,  # default to one hour
    price_per_gb_hour: float = 0.125,
    ecpu_cost: float = 0.0
) -> dict:
    gb_hours = average_gb_usage * duration_hours
    storage_cost = gb_hours * price_per_gb_hour
    total_cost = storage_cost + ecpu_cost
    return total_cost

def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
        if not file_exists:
            writer.writeheader()
        for data in dict_list:
            writer.writerow(data)

workloads = {
    'imagenet_128_nas': {
        'SHUFFLENETV2': 0.062516984,
        'RESNET18': 0.104419951,
        'RESNET50': 0.33947309,
        'VGG16': 0.514980298,
    },
    'imagenet_128_hpo': {
        'RESNET18_1': 0.104419951,
        'RESNET18_2': 0.104419951,
        'RESNET18_3': 0.104419951,
        'RESNET18_4': 0.104419951
    },
      'imagenet_128_resnet50': {
        'RESNET50': 0.33947309,
    },
     'imagenet_slowfast': {
        'SHUFFLENETV2': 0.062516984,
        'VGG16': 0.514980298,

    },

    #dict of 20 jobs operating at different speeds, labels 1-20
    '20_jobs': {
        'job_1': 0.062516984,
        'job_2': 0.104419951,
        'job_3': 0.33947309,
        'job_4': 0.514980298,
        'job_5': 0.062516984,
        'job_6': 0.104419951,
        'job_7': 0.33947309,
        'job_8': 0.514980298,
        'job_9': 0.062516984,
        'job_10': 0.104419951,
        'job_11': 0.33947309,
        'job_12': 0.514980298,
        'job_13': 0.062516984,
        'job_14': 0.104419951,
        'job_15': 0.33947309,
        'job_16': 0.514980298,
        'job_17': 0.062516984,
        'job_18': 0.104419951,
        'job_19': 0.33947309,
        'job_20': 0.514980298
    },
}


# imagenet_128_batch_size = {
#     'RESNET18': 0.104419951,
#     'RESNET50': 0.33947309,
#     'VGG16': 0.514980298,
#     'SHUFFLENETV2': 0.062516984
# }