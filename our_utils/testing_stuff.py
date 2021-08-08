from pathlib import Path
from typing import List

import numpy as np


def get_word_vector_from_model(array: np.ndarray, word_to_idx: dict, word):
    if word not in word_to_idx:
        raise KeyError(f'Word "{word}" not in vocabulary')
    vector = array[word_to_idx[word]]
    return vector / np.linalg.norm(vector)


def get_closest_n_vectors(array: np.ndarray, vector: np.array, word_to_idx, stop_words, n=1):
    word_distance_pairs = []
    for word in word_to_idx.keys():
        if word in stop_words:
            continue
        target_vector = get_word_vector_from_model(array, word_to_idx, word)
        word_distance_pairs.append((word, np.linalg.norm(vector - target_vector)))
    return sorted(word_distance_pairs, key=lambda x: x[1])[:n]


def solve_a_is_to_b_as_c_is_to(vectors: np.ndarray, word_to_idx: dict,
                               a: str, b: str, c: str, stop_words=None) -> List[str]:
    if stop_words is None:
        stop_words = set()
    vec_a = get_word_vector_from_model(vectors, word_to_idx, a)
    vec_b = get_word_vector_from_model(vectors, word_to_idx, b)
    vec_c = get_word_vector_from_model(vectors, word_to_idx, c)
    vec_d = (vec_b - vec_a) + vec_c
    vec_d /= np.linalg.norm(vec_d)
    top_10 = get_closest_n_vectors(vectors, vec_d, word_to_idx, stop_words=stop_words, n=10)
    return [x[0] for x in top_10]


def main():
    for year in range(1900, 2000, 10):
        DATA_PATH = Path('/home/jon/Documents/AdvML/final_project/data/eng-all/processed_sgns')
        with open(DATA_PATH / f'{year}-vocab.txt', 'r') as file:
            words = file.readlines()

        array = np.load(str(DATA_PATH / f'{year}-w.npy'))
        assert len(words) == array.shape[0], f'mismatch on year {year}'

        word_to_idx = dict(zip(map(lambda x: x.strip(), words), range(len(words))))
        print(year)
        query = ['ireland', 'neutral', 'england']
        print(solve_a_is_to_b_as_c_is_to(array, word_to_idx, *query,
                                         stop_words=query
                                         ))


if __name__ == '__main__':
    main()
