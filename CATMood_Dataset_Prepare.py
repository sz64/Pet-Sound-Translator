import os
import pandas as pd 

PATH_DATASETS = '../CatSound_Dataset/'
FOLDER_CSV = '../csv_output_dir/'

allowed_exts = set(['mp3', 'wav', 'au'])
column_names = ['path', 'label'] 

def write_to_csv(rows, column_names, csv_fname):
    '''rows: list of rows (= which are lists.)
    column_names: names for columns
    csv_fname: string, csv file name'''
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(os.path.join(FOLDER_CSV, csv_fname))
    
def get_rows_from_folders(folder_dataset, folders, dataroot=None):
    rows = []
    if dataroot is None:
        dataroot = PATH_DATASETS
    for label_idx, folder in enumerate(folders): # assumes different labels per folders.
        files = os.listdir(os.path.join(dataroot, folder_dataset, folder))
        files = [f for f in files if f.split('.')[-1].lower() in allowed_exts]
        for fname in files:
            file_path = os.path.join(folder_dataset, folder, fname)
            file_label = label_idx
            rows.append([file_path, file_label])
    print('Done - length:{}'.format(len(rows)))
    print(rows[0])
    print(rows[-1])
    return rows

folder_dataset_CATMood = '../CatSound_Dataset/'
labels_CATMood = ['Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind', 'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning']

n_label_CATMood = len(labels_CATMood)
folders_CATMood = [s + '/' for s in labels_CATMood]

rows_CATMood = get_rows_from_folders(folder_dataset_CATMood, folders_CATMood, '')
write_to_csv(rows_CATMood, column_names, 'CatSound_Sataset.csv')
