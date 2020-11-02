print('Top of __main__.py')

import sys
from models.char_cnn.runner import Runner

from models.char_cnn.args import get_args

# Uncomment to simulate command line arguments
# sys.argv.extend(['--dataset', 'Reuters', '--batch-size', '128', '--lr', '0.001', '--seed', '3435'])



args = get_args()
print(args)



runner = Runner(args)
runner.start()







b = 1