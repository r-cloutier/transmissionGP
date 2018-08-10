import cPickle
from egrets import *


def dump_pickle(fname, self):
    '''
    Save an input object (in this case a EGRETS) to a pickle 
    with a given file name.

    Parameters
    ----------
    `fname` : str
	The name of the output pickle to store the object.
    `self` : object (e.g. EGRETS)
	The object to save to `fname`.

    '''
    f = open(fname, 'wb')
    cPickle.dump(self, f)
    f.close()


def load_pickle(fname):
    '''
    Load a saved object in the form of a pickle.

    Parameters
    ----------
    `fname` : str
        The name of the pickle to load.

    Returns
    -------
    `self` : object (e.g. EGRETS)
        The object to stored in the pickle `fname`.

    '''
    f = open(fname, 'rb')
    self = cPickle.load(f)
    f.close()
    return self
