import data_processing as dp
import training as t
import sys

def main():
    print("Starting CS685...")

    if len(sys.argv) < 3:
        if (len(sys.argv) == 2):
            processed_filename = sys.argv[1]
        else:
            processed_filename = dp.preprocess_raw_data()
        print("Creating datasets from processed file: ", processed_filename)
        dp.create_balanced_datasets(processed_filename)
    else:
        train_filename = sys.argv[1]
        test_filename = sys.argv[2]
        t.train_datasets(train_filename, test_filename)


if __name__ == "__main__":
    main()