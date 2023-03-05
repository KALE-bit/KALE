import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            )
            xml_list.append(value)
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_names)
    return xml_df


def split_dataset(dataframe, split_ratio):
    msk = pd.DataFrame({'filename': dataframe['filename'].unique()}).sample(frac=1)
    train_index = int(len(msk) * split_ratio)
    train = msk.iloc[:train_index]
    test = msk.iloc[train_index:]
    train_df = pd.merge(dataframe, train, on='filename')
    test_df = pd.merge(dataframe, test, on='filename')
    return train_df, test_df


def main():
    path = './dataset'
    split_ratio = 0.8  # 80% for training, 20% for testing
    xml_df = xml_to_csv(path)
    train_df, test_df = split_dataset(xml_df, split_ratio)
    train_df.to_csv('train_labels.csv', index=None)
    test_df.to_csv('test_labels.csv', index=None)
    print('Successfully converted XML to CSV.')


main()
