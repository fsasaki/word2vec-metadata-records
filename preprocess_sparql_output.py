import re
import pandas as pd
umlaute_dict = {
    'ä': 'ae',  # U+00E4	   \xc3\xa4
    'ö': 'oe',  # U+00F6	   \xc3\xb6
    'ü': 'ue',  # U+00FC	   \xc3\xbc
    'Ä': 'Ae',  # U+00C4	   \xc3\x84
    'Ö': 'Oe',  # U+00D6	   \xc3\x96
    'Ü': 'Ue',  # U+00DC	   \xc3\x9c
    'ß': 'ss',  # U+00DF	   \xc3\x9f
}
def replace_german_umlaute(unicode_string):
    utf8_string = unicode_string
    for k in umlaute_dict.keys():
        utf8_string = utf8_string.replace(k, umlaute_dict[k])
    return utf8_string
def preprocess_input():
    csv_file = pd.read_csv('linked-data-query-results/result-with-delimiter.csv',header=0,delimiter=',', encoding='utf-8')
    for i, row in csv_file.iterrows():
        s = str(row.iloc[2])
        replaced1 = re.sub(' ','_',s)
        replaced2= re.sub('XXXYYY', ' ', replaced1)
        replaced3= replace_german_umlaute(replaced2)
        csv_file.loc[i]['keywords']=replaced3.lower()
    return csv_file
    pass