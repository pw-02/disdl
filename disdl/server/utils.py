from urllib.parse import urlparse
from typing import Any, Dict, List
import hashlib
import random
import numpy as np

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{val" + self.fmt +"}"
        return fmtstr.format(**self.__dict__)



def hash_list(int_list, length = 32):
    # Convert integers to strings and concatenate them
    id_string = ''.join(str(x) for x in int_list)
    
    # Hash the concatenated string to generate a unique ID
    unique_id = hashlib.md5(id_string.encode()).hexdigest()
    #Truncate the hash to the desired length
    return unique_id[:length]






def partition_dict(original_dict: Dict[Any, Any], num_partitions, batch_size=None):
   # Initialize a list to hold the partitions
    partitions = [{} for _ in range(num_partitions)]
    
    # Iterate through each key and its associated list of values in the original dictionary
    for key, values in original_dict.items():
        # Calculate the total number of values
        total_values = len(values)
        
        # Calculate the base partition size such that it is divisible by batch_size
        base_partition_size = (total_values // num_partitions) #// batch_size * batch_size
        
        # Calculate the remaining values that should be evenly distributed across partitions
        remaining_values = total_values - base_partition_size * num_partitions
        
        # Initialize the starting index for the partitions
        start_index = 0
        
        # Distribute the list of values across partitions
        for i in range(num_partitions):
            # Determine the size of the current partition
            # if i < remaining_values:
            #     # If there are remaining values, add one additional base size (batch_size) to this partition
            #     partition_size = base_partition_size + batch_size
            # else:
            #     # Otherwise, use the base partition size
            partition_size = base_partition_size
            
            # Calculate the end index for the current partition
            end_index = min(start_index + partition_size, total_values)
            
            # Get the subset of values for the current partition
            subset_values = values[start_index:end_index]
            
            # Add the key and subset of values to the current partition dictionary
            partitions[i][key] = subset_values
            
            # Update the start index for the next partition
            start_index = end_index
    
    for partition in partitions:
        partition_size = sum(len(samples) for samples in partition.values())
        if partition_size == 0:
            partitions.remove(partition)
    #calculate the total number of files across all partitions and assert that it is equal to the total number of files in the original dictionary
    total_files = sum(len(class_items) for class_items in original_dict.values())

    #print number files in each partition
    for i, partition in enumerate(partitions):
        print(f"Partition {i}: {sum(len(samples) for samples in partition.values())} files")

    assert total_files == sum(sum(len(samples) for samples in partition.values()) for partition in partitions)
    
    return partitions




# def partition_dict(original_dict: Dict[Any, Any], num_partitions, batch_size=None):
#    # Initialize a list to hold the partitions
#     partitions = [{} for _ in range(num_partitions)]
    
#     # Iterate through each key and its associated list of values in the original dictionary
#     for key, values in original_dict.items():
#         # Calculate the total number of values
#         total_values = len(values)
        
#         # Calculate the base partition size such that it is divisible by batch_size
#         base_partition_size = (total_values // num_partitions) #// batch_size * batch_size
        
#         # Calculate the remaining values that should be evenly distributed across partitions
#         remaining_values = total_values - base_partition_size * num_partitions
        
#         # Initialize the starting index for the partitions
#         start_index = 0
        
#         # Distribute the list of values across partitions
#         for i in range(num_partitions):
#             # Determine the size of the current partition
#             # if i < remaining_values:
#             #     # If there are remaining values, add one additional base size (batch_size) to this partition
#             #     partition_size = base_partition_size + batch_size
#             # else:
#             #     # Otherwise, use the base partition size
#             partition_size = base_partition_size
            
#             # Calculate the end index for the current partition
#             end_index = min(start_index + partition_size, total_values)
            
#             # Get the subset of values for the current partition
#             subset_values = values[start_index:end_index]
            
#             # Add the key and subset of values to the current partition dictionary
#             partitions[i][key] = subset_values
            
#             # Update the start index for the next partition
#             start_index = end_index
    
#     for partition in partitions:
#         partition_size = sum(len(samples) for samples in partition.values())
#         if partition_size == 0:
#             partitions.remove(partition)
#     #calculate the total number of files across all partitions and assert that it is equal to the total number of files in the original dictionary
#     total_files = sum(len(class_items) for class_items in original_dict.values())

#     #print number files in each partition
#     for i, partition in enumerate(partitions):
#         print(f"Partition {i}: {sum(len(samples) for samples in partition.values())} files")

#     assert total_files == sum(sum(len(samples) for samples in partition.values()) for partition in partitions)
    
#     return partitions
