import torch
import time
import argparse

import subprocess
import re
import random

def keep(rank):
    
    torch.cuda.set_device(rank)

    print(f'benchmarking {rank} gpus...')
    while True:
        print('rank {} gets up'.format(rank))
        # Randomly generate n between 5 and 9
        n = random.randint(5, 9)
        
        # Generate tensors a and b with dimensions 8192*n, 8192
        a = torch.rand((8192 * n, 8192)).cuda()
        b = torch.rand((8192 * n, 8192)).cuda()

        tic = time.time()
        for _ in range(500000):
            c = a * b
        toc = time.time()
        print('benchmark 500K matmul: time span: {}ms'.format((toc - tic) * 1000 / 5000))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    keep(args.gpu)