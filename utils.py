import matplotlib.pyplot as plt


def split_list_by_percentage(lst, percentage):
    # Calculate the index to split the list
    split_index = int(len(lst) * percentage / 100)

    # Split the list
    first_part = lst[:split_index]
    second_part = lst[split_index:]

    return first_part, second_part

def plot_learning_graph(learning):
    plt.plot(learning.history['loss'])
    plt.plot(learning.history['accuracy'])
    plt.title('model loss x accuracy')
    plt.xlabel('epoch')
    plt.ylabel('percentage')
    plt.legend(['train_loss', 'accuracy'], loc='upper left')

    plt.show()