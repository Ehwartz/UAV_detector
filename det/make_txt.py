import os

if __name__ == '__main__':
    path = './dataset/images'
    files = os.listdir(path)
    # f_train = open('./dataset/train.txt', 'w')

    train_txt = open('./dataset/train.txt', 'w')
    # val_txt = open('./dataset/val.txt', 'w')
    for f in files:
        train_txt.write(path + '/' + f + '\n')
        # val_txt.write(path + '/' + f + '\n')

    print('End')
