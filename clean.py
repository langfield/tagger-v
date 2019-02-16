from tqdm import tqdm

import numpy as np

import pyemblib
import array
import sys
import os 

embeddings = sys.argv[1]


def _readBin(fname, size_only=False, first_n=None, separator=' ', replace_errors=False, filter_to=None, lower_keys=False, errors='strict'):
    import sys
    words, vectors = [], []

    if filter_to:
        if lower_keys:
            filter_set = set([f.lower() for f in filter_to])
            key_filter = lambda k: k.lower() in filter_set
        else:
            filter_set = set(filter_to)
            key_filter = lambda k: k in filter_set
    else:
        key_filter = lambda k: True

    inf = open(fname, 'rb')

    # get summary info about vectors file
    summary = inf.readline().decode('utf-8', errors=errors)
    summary_chunks = [int(s.strip()) for s in summary.split(' ')]
    (numWords, dim) = summary_chunks[:2]
    if len(summary_chunks) > 2: float_size = 8
    else: float_size = 4

    if size_only:
        return (numWords, dim)

    bsep = separator.encode('utf-8')
    
    #================================
    problem_words = []
    num_errors = 0
    #================================

    for _ in tqdm(range(numWords)):
        word = []
        while True:
            next_ch = inf.read(1)
            if next_ch == b' ':
                break
            elif next_ch != b'\n':  # some files have \n separator and some do not
                word.append(next_ch)

        if replace_errors:
            word = b''.join(word).decode('utf-8', errors='replace')
        else:
            word = b''.join(word).decode('utf-8', errors=errors)
        vector = np.array(array.array('f', inf.read(dim*float_size)))

        #================================
        finite_array = np.isfinite(vector)
        nan_array = np.isnan(vector)
        if True in finite_array or True in nan_array:
            problem_words.append(key)
            num_errors += 1    
        #================================

        if key_filter(word):
            words.append(word)
            vectors.append(vector)

        if (not first_n is None) and len(words) == first_n:
            break

    inf.close()

    print("Number of errors: ", num_errors)

    # verify that we read properly
    if not first_n is None:
        assert len(words) == first_n
    elif not filter_to:
        if len(words) != numWords:
            sys.stderr.write("[WARNING] Expected %d words, read %d\n" % (numWords, len(words)))
    return (words, vectors)

words, vectors = _readBin(embeddings)



