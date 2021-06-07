import pickle


def get_pickle( pick):
    print('getting pickle : ' + pick)
    # loading
    with open( pick, 'rb') as handle:
        p = pickle.load(handle)
    return p

s = 'index_to_class.pickle'

p1 = get_pickle(s)
print (p1)
