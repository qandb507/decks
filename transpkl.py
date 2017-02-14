from pylearn2.datasets.csv_dataset import CSVDataset
import pickle

print 'convert: decks.csv -> decks.pkl'
pyln_data = CSVDataset('decks.csv',delimiter='\t',one_hot=True)
pickle.dump(pyln_data, open('decks.pkl','w'))
