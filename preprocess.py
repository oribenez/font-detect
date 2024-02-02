import cv2
import numpy as np
from PIL import Image


def process_db(db, train_mode):

    print("Stage: preproccesing data")

    im_names = list(db['data'].keys())
    list_words = []
    list_chars_fonts = []
    list_char_images = []

    # iterate images
    for curr_img in im_names:
        font = None
        img = db['data'][curr_img][:] / 255.0  # Normalize pixels
        if train_mode:
            font = db['data'][curr_img].attrs['font']
        txt = db['data'][curr_img].attrs['txt']
        char_bb = db['data'][curr_img].attrs['charBB']
        word_bb = db['data'][curr_img].attrs['wordBB']
        # print(char_bb.shape) # (2, 4, 30)
        # print(word_bb.shape) # (2, 4, 8)
        # print('Image: ', curr_img)
        # plt.imshow(img)
        # plt.show()

        char_index = 0
        # iterate words in image
        # print('Words: ', end='')
        for word in txt:
            # converting sequence of bytes to string
            word = word.decode('UTF-8')
            # print(word, end=', ')
            list_words.append(word)

            # iterate characters in word
            for char in word:
                processed_char_img = preprocess_img(
                    img, char_bb[:, :, char_index])
                # plt.imshow(processed_char_img, cmap='gray')
                # plt.show()

                if train_mode:
                    list_chars_fonts.append(font[char_index])
                list_char_images.append(processed_char_img)
                char_index += 1

    return list_words, list_chars_fonts, list_char_images


def preprocess_img(img, char_bb):
    # warp perspective
    # the exact resolution will be (128,128) after adding margins
    dest_res_bb = (108, 108)
    h_margin = 10
    v_margin = 10

    src_bb = char_bb.transpose()
    # 2D plane bounding box (topleft, topright, bottomright, bottomleft)
    dest_bb = np.array([[h_margin, v_margin],
                        [h_margin + dest_res_bb[0], v_margin],
                        [h_margin + dest_res_bb[0], v_margin + dest_res_bb[1]],
                        [h_margin, v_margin + dest_res_bb[1]]])
    # resolution of the target concentrated image (including margins)
    dest_img_res = (2*h_margin + dest_res_bb[0], 2*v_margin + dest_res_bb[1])

    # Find the homography matrix
    # H, _ = ...    is a tuple while first arg is going to 'H' var and second, '_', is ignored
    H, _ = cv2.findHomography(src_bb, dest_bb)

    # Warp the source image using the homography matrix
    dst_image = cv2.warpPerspective(img, H, dest_img_res)

    # Convert image to grayscale
    dst_image = cv2.cvtColor(np.float32(dst_image), cv2.COLOR_BGR2GRAY)

    dst_image = Image.fromarray(dst_image)
    dst_image = np.array(dst_image.resize((32, 32)))

    return dst_image
