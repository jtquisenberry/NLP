# python -m models.char_cnn --dataset Reuters --batch-size 128 --lr 0.001 --seed 3435

# from .models import char_cnn
import models.char_cnn

import sys
sys.argv.append('abcdef')


from models.char_cnn import __main__

print('xxxxxxxxx')


a = 1






