import pandas as pd
import datetime
from datasets import Features, Value, ClassLabel, Dataset

def preprocess_raw_data():
    print("PreProcess raw data...")

    print("Read discharge notes...")
    df_discharge = pd.read_csv("./data/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz")

    print("Read diagnosis codes...")
    df_diag = pd.read_csv("./data/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz")

    print("Filter any codes that are not ICD-10...")
    df_diag = df_diag.loc[df_diag['icd_version'] == 10]

    print("Create combined unique id for discharge notes...")
    df_discharge['subject_hadm_id'] = df_discharge['subject_id'].astype(str) + '_' + df_discharge['hadm_id'].astype(str)

    print("Create combined unique id for diagnosis codes...")
    df_diag['subject_hadm_id'] = df_diag['subject_id'].astype(str) + '_' + df_diag['hadm_id'].astype(str)

    print("Filter out diagnosis codes that do not have notes...")
    df_diag = df_diag.loc[df_diag['subject_hadm_id'].isin(df_discharge['subject_hadm_id'].unique())].copy()

    print("Group all diagnosis codes by unique id...")
    df_diag_grouped = df_diag.groupby(['subject_hadm_id'])

    def calc_icdcode(value):
        return ', '.join(df_diag_grouped.get_group(value)['icd_code'].tolist())

    print("Add column for grouped diagnosis codes...")
    df_diag['icd_code_all'] = df_diag['subject_hadm_id'].map(calc_icdcode)

    print("Collapse all of the diagnosis rows to one per unique id...")
    df_diag = df_diag.groupby(['subject_id', 'hadm_id', 'subject_hadm_id', 'icd_code_all'], as_index=False).agg({'seq_num':'min'})

    print("Join notes with discharge codes...")
    df_discharge_icd10_joined = pd.merge(df_diag, df_discharge, on=['subject_hadm_id'], how="left")

    print("Remove unnecessary columns...")
    df_discharge_icd10_joined.drop(columns=['subject_id_x', 'hadm_id_x', 'seq_num', 'note_id', 'subject_id_y', 'hadm_id_y', 'note_type', 'note_seq', 'charttime', 'storetime'], inplace=True)

    datetime.date.today().strftime("%Y%m%d")

    filename = './data/discharge_icd10_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.csv.gz'

    print('Write data to: ', filename)
    df_discharge_icd10_joined.to_csv(filename, index=False)

    print("Done pre-processing raw data.")

    return filename

def create_balanced_datasets(filename):
    print("Read preprocessed data file into DataFrame...")
    df = pd.read_csv(filename, index_col=0)

    def calc_cancer(value):
        if 'C' in value:
            return '1'
        else:
            return '0'

    print("Add label column for Cancer/NoCancer (1/0)...")
    df['label'] = df['icd_code_all'].map(calc_cancer)

    print("Drop unecessary columns...")
    df.drop(columns=['icd_code_all'], inplace=True)

    def write_split_dataset_sample(dforiginal, sample_size, features):
        print("Create datasets for sample size ", sample_size, '...')
        dfsample = dforiginal.groupby('label', group_keys=False).apply(lambda x: x.sample(n=sample_size))
        print("Shuffle data...")
        dfsample = dfsample.sample(frac=1)
        print("Create train/test split...")
        train = dfsample[sample_size:].sample(frac=0.7)
        test = dfsample[:sample_size].sample(frac=0.3)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        train_dataset = Dataset.from_pandas(train, split="train", features=features)
        test_dataset = Dataset.from_pandas(test, split="test", features=features)
        train_file = "./data/train_" + str(sample_size) + ".hf"
        print("Save dataset to disk: ", train_file)
        train_dataset.save_to_disk(train_file)
        test_file = "./data/test_" + str(sample_size) + ".hf"
        print("Save dataset to disk: ", test_file)
        test_dataset.save_to_disk(test_file)

    icd_features = Features({'text': Value('large_string'), 'label': ClassLabel(names=['0', '1'])})

    write_split_dataset_sample(df, 100, icd_features)
    write_split_dataset_sample(df, 1000, icd_features)
    write_split_dataset_sample(df, 10000, icd_features)

    min_label = min(df.label.value_counts()[0], df.label.value_counts()[1])
    write_split_dataset_sample(df, min_label, icd_features)