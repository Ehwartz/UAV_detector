import os


def rename(path):
    img_list = os.listdir(path)

    for i, img_name in enumerate(img_list):
        os.rename(path + '/' + img_name, path + '/' + '%d.jpg' % i)


if __name__ == '__main__':
    # rename(r'D:/Downloads/screw')
    print(os.listdir(r'D:/Downloads/screw'))
    print('End')
