"""
Class for managing our data.
"""
import csv
import glob
import operator
import os.path
import random
import threading
from typing import Counter

import numpy as np
from keras.utils import to_categorical
from processor import process_image


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():
    """Class to handle UCF-101 dataset."""
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """
        Constructor for dataset.

        Parameters:
            seq_length (`int`): The number of frames to consider
            class_limit (`int`): Number of classes to limit the data to.
            image_shape (`tuple`): The shape of the input images
        
        Returns:
            An instance of the DataSet class
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = os.path.join('data', 'sequences')
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data() -> list[list[str]]:
        """
        Loads data from `data_file.csv`.\n
        Csv is is in format `[train/test, class, filename, num_frames]`.\n
        Each list item is one sample descibed in the way above.

        Returns:
            list[list[str]]: The dataset
        """
        with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """
        Excludes samples that are shorter than `self.seq_length` frames or longer
        than `self.max_frames` frames. Also limits samples to classes in
        `self.classes`.

        Returns:
            list[list[str]]: The cleaned dataset
        """
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """
        Extract the classnames in data. 
        Amount of classes is limited by setting `self.class_limit`.
        
        Returns:
            list[str]: The list of class names
        """
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            # return classes[:self.class_limit] # when having not all class names available
            return classes[::-1][:self.class_limit] # when having all class names available, do flip to start at end
        else:
            return classes

    def get_class_one_hot(self, class_str: str):
        """
        Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training.
        
        Parameters:
            class_str (`str`): The class name as a string
        
        Returns:
            np.ndarray: The one-hot encoded class
        """
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_test(self):
        """
        Split the data into train and test groups.
        belonging to train and test is determined by the first column in self.data.

        Returns:
            tuple: (train data, test data)
        """
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test: str, data_type: str):
        """
        This is a mirror of the generator, but attempts to load everything into
        memory so we can train way faster.

        Parameters:
            train_test (`str`): 'train' or 'test' data to load
            data_type (`str`): Should the actual image ("images") or or retrieved features ("features") be used?

        Returns:
            tuple: A tuple of (X, y) where X is the input data and y are the labels
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:

            if data_type == 'images':
                frames = self.get_frames_for_sample(row)
                frames = self.rescale_list(frames, self.seq_length)

                # Build the image sequence
                sequence = self.build_image_sequence(frames)

            else:
                sequence = self.get_extracted_sequence(data_type, row)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, batch_size: int, train_test: str, data_type: str):
        """
        Return a generator that we can use to train on. Loops indefinitely.

        Parameters:
            batch_size (`int`): The batch size to use
            train_test (`str`): 'train' or 'test' generator
            data_type (`str`): Should the actual image ("images") or or retrieved features ("features") be used?

        Yields:
            tuple: A tuple of (X, y) where X is the input data and y are the labels
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type == "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames, self.seq_length)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                    if sequence is None:
                        raise ValueError("Can't find sequence. Did you generate them?")

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames: list[str]) -> list[np.ndarray]:
        """
        Given a set of frames (filenames), build the image sequence by loading it as numpy array and normalizing it.
        Provides a list of sorted processed images.
        
        Parameters:
            frames (`list[str]`): The list of frame filenames
        
        Returns:
            list[np.ndarray]: The list of processed images        
        """
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type: str, sample: list[str]) -> np.ndarray:
        """
        Function for getting the saved extracted features for a sample (after applying InceptionV3 CNN).
        Get the saved extracted features by creating the path to the needed feature with `.npy` ending.

        Parameters:
            data_type (`str`): The type of data to get (e.g. 'features')
            sample (`list[str]`): A sample row from the data file (structure: `[train/test, class, filename, num_frames]`)
        
        Returns:
            np.ndarray: The extracted feature sequence or None if not found        
        """
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-' + data_type + '.npy')
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename: str, data_type: str):
        """
        Given a filename for one of our samples, return the data
        the model needs to make predictions.
        
        Parameters:
            filename (`str`): The filename of the sample (without path or extension)
            data_type (`str`): The type of data to get (either 'images' or 'features')
            
        Returns:
            The data sequence for the sample (either image sequence or extracted features)
        """
        # First, find the sample row.
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)

        if data_type == "images":
            # Get and resample frames.
            frames = self.get_frames_for_sample(sample)
            frames = self.rescale_list(frames, self.seq_length)
            # Build the image sequence
            sequence = self.build_image_sequence(frames)
        else:
            # Get the sequence from disk.
            sequence = self.get_extracted_sequence(data_type, sample)

            if sequence is None:
                raise ValueError("Can't find sequence. Did you generate them?")

        return sequence

    @staticmethod
    def get_frames_for_sample(sample: list[str]) -> list[str]:
        """
        Given a sample row from the data file, get all the corresponding frame
        filenames.
        
        Parameters:
            sample (`list[str]`): A sample row from the data file (structure: `[train/test, class, filename, num_frames]`)

        Returns:
            list[str]: The list of frame filenames for the video sample        
        """
        path = os.path.join('data', sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split(os.path.sep)
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """
        Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the original list.
        
        Parameters:
            input_list (`list`): The list to rescale
            size (`int`): The desired size of the output list

        Returns:
            list: The rescaled list
        """
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(self.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))

    def plot_dataset(self):
        """
        Creates a plot summary of the dataset including:
        - Histogram of frame counts per video
        - Boxplot of frame counts by train/test split
        - Bar chart of video counts per class
        - Bar chart of average frame counts for top 10 classes
        """
        mode, class_name, _, nb_frames = zip(*self.data)

        mean_frames = np.mean([int(x) for x in nb_frames])
        median_frames = np.median([int(x) for x in nb_frames])
        class_counts = Counter(class_name)
        
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sns.set_style("whitegrid")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist([int(x) for x in nb_frames], bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(mean_frames, color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].axvline(median_frames, color='darkred', linestyle='--', linewidth=2, label='Median')
        axes[0, 0].set_xlabel('Number of Frames')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Frame Distribution per Video')
        axes[0, 0].legend()

        sns.boxplot(x=mode, y=[int(x) for x in nb_frames], ax=axes[0, 1], palette="pastel")
        axes[0, 1].set_xlabel('Split')
        axes[0, 1].set_ylabel('Number of Frames')
        axes[0, 1].set_title('Frame Distribution by Split')
        plt.sca(axes[0, 1])
        plt.xticks(rotation=0)

        class_count_series = pd.Series(class_counts).sort_values(ascending=True)
        axes[1, 0].barh(class_count_series.index, class_count_series.values, color='lightgreen', edgecolor='black')
        axes[1, 0].set_yticks(class_count_series.index)
        axes[1, 0].set_yticklabels(class_count_series.index, fontsize=7)
        axes[1, 0].set_xlabel('Number of Videos')
        axes[1, 0].set_title('Video Distribution by Class')
        axes[1, 0].axvline(np.mean(class_count_series.values), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1, 0].legend()

        df = pd.DataFrame({
            'class': class_name,
            'nb_frames': [int(x) for x in nb_frames]
        })
        class_frames = df.groupby('class')['nb_frames'].mean().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(class_frames)), class_frames.values, color='coral', alpha=0.7)
        axes[1, 1].set_yticks(range(len(class_frames)))
        axes[1, 1].set_yticklabels(class_frames.index, fontsize=9)
        axes[1, 1].set_xlabel('Average Frames')
        axes[1, 1].set_title('Top 10 Classes by Average Frame Count')
        axes[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.show()
