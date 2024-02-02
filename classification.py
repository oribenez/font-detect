import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import load_model
from collections import Counter

from config import __DIRNAME_INPUT__, __DIRNAME_WORKING__, __DIRNAME_OUTPUT__, __FONTS_DICT__

def submit_predictions(fonts_prediction_ids):

    df = pd.DataFrame(fonts_prediction_ids)
    df['Index'] = range(len(df))
    new_df = pd.DataFrame({'ind': df['Index'], 'font': fonts_prediction_ids})
    new_df.to_csv(__DIRNAME_OUTPUT__ + "submission.csv", index=False)

# imporve predictions based on knowledge regarding the database


def majority_vote(list_words, predict_fonts):
    num_fonts = len(__FONTS_DICT__)

    new_predict_fonts = []
    count_changes = 0
    i = 0
    for word_ind, word in enumerate(list_words):
        curr_word_votes = predict_fonts[i:i+len(word)]

        # Count votes for each font
        vote_counts = Counter(curr_word_votes)
        vote_decision = vote_counts.most_common(1)[0][0]

        # change prediction of font for all chars in word so they all have the same font and best decision
        flag_word_font_prediction_changed = False
        for ch_ind, char in enumerate(word):
            if curr_word_votes[ch_ind] != vote_decision and not flag_word_font_prediction_changed:
                count_changes += 1

            new_predict_fonts.append(vote_decision)

        i += len(word)

    print('count_changes_on_words: ', count_changes, ' / ', len(list_words))

    return new_predict_fonts


def predict(list_words, list_char_images, list_chars_fonts, filename="best_model.h5"):

    trained_model = load_model(__DIRNAME_WORKING__ + filename)
    predictions_list = trained_model.predict(np.array(list_char_images))

    predict_fonts = []
    for i in range(len(predictions_list)):
        im = list(predictions_list[i])
        predict_fonts += [im.index(max(im))]

    # Fact: every word has only one font. means that there is no word built of couple fonts for every character.
    # based on the above fact we can improve the prediction by using majority vote.
    print("Stage: imporving prediction by knowledge on the given data")
    print('# Fact: every word has only one font. means that there is no word built of couple fonts for every character.')
    predict_fonts = majority_vote(list_words, predict_fonts)

    # show more data on the predictions that has been made. relevant only if we have the real-world labels
    data_has_realworld_labels = len(list_chars_fonts) > 0
    if data_has_realworld_labels:
        # Reverse the original dictionary for mapping names to numbers
        # {'Flower Rose Brush': 0, 'Skylark': 1, 'Sweet Puppy': 2, 'Ubuntu Mono': 3, 'VertigoFLF': 4, 'Wanted M54': 5, 'always forever': 6}
        reverse_fonts_dict = {v: k for k, v in __FONTS_DICT__.items()}
        # Map font names to numbers
        list_chars_fonts_id = [reverse_fonts_dict[name]
                               for name in list_chars_fonts]

        count_correct_predictions = 0
        for i in range(len(list_chars_fonts_id)):
            if list_chars_fonts_id[i] == predict_fonts[i]:
                count_correct_predictions += 1

        print("# based on validation data:")
        print('count_correct_predictions: ', count_correct_predictions,
              ' / ', len(list_chars_fonts_id))
        print(len(list_chars_fonts_id))
        print('accuracy: ', (count_correct_predictions/len(list_chars_fonts_id))*100)
        print()
    return predict_fonts
