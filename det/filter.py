import xml
import os

classes = ['cookies', 'mixed_congee', 'tomato_sauce', 'chocolate', 'melon_seeds',
        'milk', 'water', 'cola', 'coffee', 'ad_milk',
        'knife', 'underwear_detergent', 'book', 'floral_water', 'toothpaste',
        'folder', 'water_glass', 'food_grade_detergent', 'slippers', 'pen']


def file_filter(path):
    files = os.listdir(path)
