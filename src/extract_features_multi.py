"""
Parallel feature extraction for videos using InceptionV3
"""

import os
from multiprocessing import Pool

import numpy as np
from data import DataSet
from extractor import Extractor
from tqdm import tqdm

# Configuration
seq_length = 40
class_limit = 40
num_workers = 3

def init_worker():
    """Each worker loads its own extractor."""
    global model
    model = Extractor()

def process_video(video):
    """Process one video sample."""
    path = os.path.join('data', 'sequences', f"{video[2]}-{seq_length}-features.npy")

    if os.path.isfile(path):
        return video[2]

    # for safety reasons import DataSet again to have it in each worker scope
    from data import DataSet
    data = DataSet(seq_length=seq_length, class_limit=class_limit)

    frames = data.get_frames_for_sample(video)
    frames = data.rescale_list(frames, seq_length)

    sequence = [model.extract(image) for image in frames]
    np.save(path, sequence)
    return video[2]

if __name__ == "__main__":
    data = DataSet(seq_length=seq_length, class_limit=class_limit)
    all_videos = data.data

    print(f"Starting parallel feature extraction with {num_workers} workers...")
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for _ in tqdm(pool.imap_unordered(process_video, all_videos), total=len(all_videos)):
            pass

    print("Feature extraction complete.")