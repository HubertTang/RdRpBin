import cnn_dataset
import cnn_model
import numpy as np
import pandas as pd
import subprocess
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pred_fc(data_loader, model, num_hidden, num_data):
    """Generate the embedding input vector using the trained model.
    """
    model.eval()
    result = np.ones((num_data, num_hidden))

    before_index = 0

    for (datas, labels) in tqdm(data_loader):
        num_s = datas.shape[0]
        datas = datas.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(datas)

            temp_npy = outputs.cpu().numpy()

            result[before_index: before_index + num_s, :] = temp_npy
            before_index += num_s

    return result


def run_CNN_pred(database_name, test_csv, num_class=18, seq_len=66, 
                 num_token=21, batch_size=1024, threads=1):
    """Run CNN and output the FC1 vector.
    """
    # from history.csv import the parameters of the model
    df = pd.read_csv("CNN_param.log", sep=',')

    for index, n in enumerate(df['Name']):
        if n == database_name:
            row_id = index
            break

    filter_size = [int(i) for i in df['Filter_size'][row_id].strip().strip('[]').split()]
    num_filter = int(df['Num_filters'][row_id])
    dropout = float(df['Dropout'][row_id])
    num_hidden = int(df['Num_hidden'][row_id])

    # load the testing dataset
    test_data = cnn_dataset.PepOnehot(file=test_csv, seq_len=seq_len)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=threads)

    # load the trained model
    trained_model = cnn_model.RdRpBinCNN(num_class=num_class, num_token=num_token, 
                                         seq_len=seq_len, kernel_nums=num_filter, kernel_sizes=filter_size, 
                                         dropout=dropout, num_fc=num_hidden, out_logit=False).to(device)
    trained_model.load_state_dict(torch.load(f"{database_name}/cnn_model.pt"))
    
    num_test = int(subprocess.check_output(f'wc -l {test_csv}', shell=True).split()[0])

    temp_arr = pred_fc(test_loader, trained_model, num_hidden, num_test)
    print(f"Test embedding vector size: {temp_arr.shape}")
    np.save(f"{test_csv}.FC1", temp_arr)


