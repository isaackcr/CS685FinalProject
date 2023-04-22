import data_processing as dp
import training as t
import sys

def main():
    print("Starting CS685...")

    # Check command line args
    if len(sys.argv) < 3:
        # If args is 2, a preprocessed file was passed in, and we need to create new randomized datasets from it
        if (len(sys.argv) == 2):
            processed_filename = sys.argv[1]
        else:
        # If no args were passed in, we need to start from scratch by preprocessing the raw data
            processed_filename = dp.preprocess_raw_data()
        print("Creating datasets from processed file: ", processed_filename)
        dp.create_balanced_datasets(processed_filename)
    #If args is 3, then the user passed in a training dataset and a testing dataset and wants to fine-tune a model
    else:
        train_filename = sys.argv[1]
        test_filename = sys.argv[2]
        t.fine_tune(train_filename, test_filename)

if __name__ == "__main__":
    main()