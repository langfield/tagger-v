from tqdm import tqdm

import numpy as np

import pyemblib
import array
import sys
import os 

target_list_path = sys.argv[1]

#========1=========2=========3=========4=========5=========6=========7==

class Embeddings(dict):
    '''Wrapper for word embeddings; inherits from Dictionary.
    Keys are words, values are embedding arrays.
    '''
    @property
    def size(self):
        if not hasattr(self, '_size'):
            for any_vector in self.values():
                break
            self._size = len(any_vector)
        return self._size
    @property
    def dimension(self):
        return self.size
    @property
    def shape(self):
        return (len(self), self.size)

    def has(self, key):
        return not self.get(key, None) is None

    def toarray(self, ordered=False):
        '''Returns the embedding vocabulary in fixed order and
        a NumPy array of the embeddings, in vocab order.
        '''
        if ordered:
            vocab = list(self.keys())
            vocab.sort()
            vocab = tuple(vocab)
        else:
            vocab = tuple(self.keys())
        embed_array = []
        for v in vocab: embed_array.append(self[v])
        return (vocab, numpy.array(embed_array))

#========1=========2=========3=========4=========5=========6=========7==

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
        if False in finite_array or True in nan_array:
            problem_words.append(word)
            num_errors += 1    
        #================================

        elif key_filter(word):
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

#========1=========2=========3=========4=========5=========6=========7==

def loopflow(target_list_path):
    with open(target_list_path) as f:
        target_list = f.readlines()

    for i,target in enumerate(target_list):
        target = os.path.abspath(target)
        target = list(target)
        target.remove('\n')
        target = "".join(target)
        basename = os.path.basename(target)
        parent = os.path.abspath(os.path.join(target, '../'))
        extension = target.split('.')[-1]
        os.system('srun -J 15GB-4c --mem 15000 -c 4 -w zirconium python3 /u/user/tagger-v/clean.py ' + target)

        words, vectors = _readBin(target)

        lower_keys = False
        wordmap = Embeddings()
        for i in range(len(words)):
            if lower_keys: key = words[i].lower()
            else: key = words[i]
            wordmap[key] = vectors[i]

        save_name = os.path.join(parent,'parse-error-fix_' + basename + '.' + extension)
        pyemblib.write(wordmap, save_name, mode=pyemblib.Mode.Binary)

if __name__ == '__main__':
    loopflow(target_list_path)
