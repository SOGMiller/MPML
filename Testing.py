import mpml2 as mp

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import pandas as pd
import pathlib

path = str(pathlib.Path(__file__).parent.absolute())

algo = GaussianNB()
dataFrame = pd.read_csv (path + '/botnet_train3.csv')

mpml = mp.mpml(algo,dataFrame,"class")    

# mpml.calc_sig()
# mpml.gen_relation_scores()
# mpml.gen_perspectives() 
# mpml.analyse_inst(88)

mpml.list_features()

per = [['avg_byte', 'var_byte', 'protocol', 'pack_exc', 'byte_exc'],[ 'pack_push', 'percent_push', 'src_port', 'dst_port', 'std_byte'],['percent_push', 'src_port', 'dst_port', 'std_byte']]

mpml.gen_custom_perspectives(per)

# mpml.view_perspectives()

instList = [28,29,54,55]

mpml.analyse_inst_list(instList)