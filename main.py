import torch
import os
from datetime import datetime
from SemiLT.trainingprocess import Training
from SemiLT.transfer import Transfer
import time
from setting import Setting
import random
random.seed(1)

def main():
    # hardware constraint for speed test
    start_time = time.time()
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # initialization 
    setting = Setting()    
    torch.manual_seed(setting.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))
    
    # Training
    print('SemiLT start:')
    model_stage1= Training(setting)    
    for epoch in range(setting.epochs):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('SemiLT finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # Label transfer
    print('Label transfer:')
    Transfer(setting, neighbors = 10, knn_rna_samples=50000)
    print('Label transfer finished: ', datetime.now().strftime('%H:%M:%S'))
    
    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    print(f"Run time：{hours}: {minutes}: {seconds}")
    
if __name__ == "__main__":
    main()