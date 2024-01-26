import os
from xml.etree import ElementTree as ET
from xml.dom import minidom


def get_class(path):
    filelist = os.listdir(path)
    print(filelist)
    classes = list()

    for f in filelist:
        tree = minidom.parse(path+'/'+f)
        print(tree)
        root = tree.documentElement
        print(root.childNodes)
        print(root.hasAttribute('annotation'))
        objs = root.getElementsByTagName('annotation')[0].getElementsByTagName('object')

        for obj in objs:
            name = obj.getAttribute('name')
            print(name)


if __name__ == '__main__':
    get_class('D:/Downloads/Annotations')
