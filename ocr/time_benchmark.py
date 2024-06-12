import time

import numpy as np
import pandas as pd
from text_extractor import TextExtractor
from tqdm import tqdm

if __name__ == "__main__":
    num_links = 50
    df = pd.read_csv("../data/yappy_hackaton_2024_400k.csv")
    text_extractor = TextExtractor()

    def measure_time_execution(video_link):
        start_time = time.time()
        texts, num_frames = text_extractor.extract_text(video_link=video_link)
        unique_texts = list(dict.fromkeys(texts))
        end_time = time.time()
        print(video_link, unique_texts)
        return (
            round(end_time - start_time, 2),
            len(texts),
            len(unique_texts),
            num_frames,
            len(unique_texts) / num_frames,
        )

    results = np.array([0.0] * 5)
    links = df.sample(num_links, random_state=42).link.tolist()

    for link in tqdm(links):
        stats = measure_time_execution(link)
        print(stats)
        results += np.array(stats) / len(links)

    print(
        f"Number of measures: {num_links}\n"
        f"Average Time: {results[0]}\n"
        f"Average Number of texts: {results[1]}\n"
        f"Average Number of unique texts: {results[2]}\n"
        f"Average Number of frames: {results[3]}\n"
        f"Аverage ratio of unique texts to the number of frames: {results[4]}"
    )
