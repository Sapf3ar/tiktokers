"""
This module measures the execution time of the text extraction process from a video.

The module reads a CSV file containing video links, extracts text from each video using
the TextExtractor class, and measures the time taken for each extraction.
It then calculates and prints the average time taken, the average number of texts extracted,
and the average number of unique texts extracted.

Functions
---------
measure_time_execution(video_link)
    Measures the time taken to extract text from a video.
"""

import time

import numpy as np
import pandas as pd
from text_extractor import TextExtractor
from tqdm import tqdm

if __name__ == "__main__":
    NUM_LINKS = 50
    df = pd.read_csv("../data/yappy_hackaton_2024_400k.csv")
    text_extractor = TextExtractor()

    def measure_time_execution(video_link):
        """
        Measures the time taken to extract text from a video.

        This function extracts text from a video, measures the time taken for the extraction,
        and returns the time taken, the number of texts extracted,
        and the number of unique texts extracted.

        Parameters
        ----------
        video_link : str
            The link to the video.

        Returns
        -------
        tuple
            A tuple containing the time taken for the extraction (in seconds),
            the number of texts extracted, and the number of unique texts extracted.
        """
        start_time = time.time()
        texts = text_extractor.extract_text(video_link=video_link)
        unique_texts = list(dict.fromkeys(texts))
        end_time = time.time()
        print(video_link, unique_texts)
        return (round(end_time - start_time, 2), len(texts), len(unique_texts))

    results = np.array([0.0] * 3)
    links = df.sample(NUM_LINKS, random_state=42).link.tolist()

    for link in tqdm(links):
        stats = measure_time_execution(link)
        print(stats)
        results += np.array(stats) / len(links)

    print(
        f"Number of measures: {NUM_LINKS}\n"
        f"Average Time: {results[0]}\n"
        f"Average Number of texts: {results[1]}\n"
        f"Average Number of unique texts: {results[2]}\n"
    )
