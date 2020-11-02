# python -m models.char_cnn --dataset Reuters --batch-size 128 --lr 0.001 --seed 3435

# from .models import char_cnn
#import models.char_cnn

import sys
# sys.argv.append('abcdef')
#sys.argv.extend(['--dataset', 'Reuters', '--batch-size', '128', '--lr', '0.001', '--seed', '3435'])
#sys.argv.append('3')



from models.char_cnn.runner import Runner
from models.char_cnn.runner import CustomArgs
args = CustomArgs()

args.dataset='Reuters'
args.batch_size=128
args.lr=0.001
args.seed=3435

print(args)



runner = Runner(args)
runner.start()





