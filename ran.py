import sys
import os

target_list_path = sys.argv[1]

with open(target_list_path) as f:
    target_list = f.readlines()

for i,target in enumerate(target_list):
    target = os.path.abspath(target)
    target = list(target)
    target.remove('\n')
    target = "".join(target)
    basename = os.path.basename(target)
    extension = target.split('.')[-1]
    # os.system('srun -J googpred -w vibranium --mem 20000 -c 4 python3 experiment.py --dataset Google --predictions-file data/logs/Google.' + basename + '.predictions -l data/logs/Google.' + basename + '.log ' + '--embeddings ' + target)
    # script to run with pretrained embedding (dist embedding)
    os.system('srun -J 10GB-4c --mem 10000 -c 4 -w adamantium python2 /u/user/tagger-v/train.py --word_dim=300 --train=dataset/eng.train --dev=dataset/eng.testa --test=dataset/eng.testb --pre_emb=' + target)

