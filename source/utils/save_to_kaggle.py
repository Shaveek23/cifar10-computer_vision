import pandas as pd



def save_to_kaggle(datset_size, output, file_name):
    data  = pd.DataFrame(columns = ['id','label'])
    data['id'] = list(range(1,datset_size+1))
    data['label'] = output
    data.to_csv(f"{file_name}.csv", sep=',', index = False)

    