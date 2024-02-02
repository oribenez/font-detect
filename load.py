import h5py

from classification import predict, submit_predictions
from config import __DIRNAME_INPUT__, __FONTS_DICT__
from preprocess import process_db
from training import create_model, train_model
from utils import split_list_by_percentage



def load_training_and_validation(ds_filename='train.h5'):
    print("Stage: load training and validation database")

    path_input = __DIRNAME_INPUT__ + ds_filename
    with h5py.File(path_input, 'r') as db:
        # process training database
        list_words, list_chars_fonts, list_char_images = process_db(
            db, train_mode=True)

        # split data to validation and training
        print("Stage: split data to validation and training")
        amount_training_data = 80  # percentage value # %
        list_words_training, list_words_validation = split_list_by_percentage(
            list_words, amount_training_data)

        # get num chars in training set
        num_chars_in_trainning = 0
        for word in list_words_training:
            for char in word:
                num_chars_in_trainning += 1

        list_char_images_training, list_char_images_validation = list_char_images[
            :num_chars_in_trainning], list_char_images[num_chars_in_trainning:]
        list_chars_fonts_training, list_chars_fonts_validation = list_chars_fonts[
            :num_chars_in_trainning], list_chars_fonts[num_chars_in_trainning:]

        print("list_words_validation: ", len(list_words_validation))
        # create model for font detection
        new_model = create_model(len(__FONTS_DICT__))

        # train the model
        print("Stage: training")
        trained_model = train_model(new_model, list_char_images_training, list_chars_fonts_training,
                                    list_char_images_validation, list_chars_fonts_validation)

        # predict fonts from images on validation data
        print("Stage: making predictions on validation data")
        fonts_prediction_ids = predict(
            list_words_validation, list_char_images_validation, list_chars_fonts_validation)


def predict_on_test_data(classification_model_file_name, ds_filename='test.h5'):
    print("Stage: load test database")

    path_input = __DIRNAME_INPUT__ + ds_filename
    with h5py.File(path_input, 'r') as db:
        # process test database
        list_words, list_chars_fonts, list_char_images = process_db(
            db, train_mode=False)

        # predict fonts from images on test data
        print("Stage: making predictions on test data")
        fonts_prediction_ids = predict(
            list_words, list_char_images, list_chars_fonts, classification_model_file_name)

        # submit predictions
        print("Stage: save fonts predictions as .csv file")
        submit_predictions(fonts_prediction_ids)
