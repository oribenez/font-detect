import sys
from config import models

from load import load_training_and_validation, predict_on_test_data

def main():
    try:
        choice = input('Please choose one of the following options:\n1. Perform classifications on the file `test.h5` based on the best trained model\n2. Train a new model, make predictions on file test.5\n')
        
        if choice == '1':
            classification_model_file_name = models.BEST_MODEL
        elif choice == '2':
            load_training_and_validation()
            classification_model_file_name = models.LAST_TRAINED_MODEL
        else:
            print('please choose the correct answer. next time... bye')
            sys.exit()

        # make predictions on a specified  trained model for the test data
        predict_on_test_data(classification_model_file_name)

    except KeyboardInterrupt:
        print('App is closing. bye bye...')
        sys.exit()
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == '__main__':
    main()
