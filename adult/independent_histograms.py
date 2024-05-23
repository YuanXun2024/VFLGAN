from pandas import DataFrame
from numpy import ndarray, array, linspace, all, nanmean, nanmedian, nanvar
from pandas.api.types import CategoricalDtype
import json
from pandas.api.types import is_numeric_dtype, CategoricalDtype

from warnings import filterwarnings
filterwarnings('ignore', message=r"Parsing", category=FutureWarning)


# Data coding constants
FILLNA_VALUE_CAT = "NaN"
CATEGORICAL = "Categorical"
ORDINAL = "Ordinal"
INTEGER = "Integer"
FLOAT = "Float"
NUMERICAL = [INTEGER, FLOAT]
STRINGS = [CATEGORICAL, ORDINAL]

# Runtime constant
PROCESSES = 16

# Experiment constants
LABEL_IN = 1
LABEL_OUT = 0
ZERO_TOL = 1e-12


class FeatureSet(object):
    def extract(self, data):
        return NotImplementedError('Method needs to be overwritten by subclass')


class NaiveFeatureSet(FeatureSet):
    def __init__(self, datatype):
        self.datatype = datatype
        self.attributes = None
        self.category_codes = {}
        assert self.datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)

        self.__name__ = 'Naive'

    def extract(self, data):
        if self.datatype is DataFrame:
            assert isinstance(data, DataFrame), 'Feature extraction expects DataFrame as input'
            if self.attributes is not None:
                if bool(set(list(data)).difference(set(self.attributes))):
                    raise ValueError('Data to filter does not match expected schema')
            else:
                self.attributes = list(data)
            features = DataFrame(columns=self.attributes)
            for c in self.attributes:
                col = data[c]
                if is_numeric_dtype(col):
                    features[c] = [col.mean(), col.median(), col.var()]
                else:
                    if c in self.category_codes.keys():
                        new_cats = set(col.astype('category').cat.categories).difference(set(self.category_codes[c]))
                        self.category_codes[c] += list(new_cats)
                        col = col.astype(CategoricalDtype(categories=self.category_codes[c]))
                    else:
                        col = col.astype('category')
                        self.category_codes[c] = list(col.cat.categories)
                    counts = list(col.cat.codes.value_counts().index)
                    features[c] = [counts[0], counts[-1], len(counts)]
            features = features.values

        elif self.datatype is ndarray:
            assert isinstance(data, ndarray), 'Feature extraction expects ndarray as input'
            features = array([nanmean(data), nanmedian(data), nanvar(data)])
            # features = array([nanmean(data, axis=0), nanmedian(data, axis=0), nanvar(data, axis=0)])
        else:
            raise ValueError(f'Unknown data type {type(data)}')

        return features.flatten()


class HistogramFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=45, quids=None):
        assert datatype in [DataFrame], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attributes = []
        self.num_attributes = []

        self.histogram_bins = {}
        self.category_codes = {}

        if quids is None:
            quids = []

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            dtype = cdict['type']

            if dtype == FLOAT or dtype == INTEGER:
                if attr_name not in quids:
                    self.num_attributes.append(attr_name)
                    self.histogram_bins[attr_name] = linspace(cdict['min'], cdict['max'], nbins+1)
                    self.nfeatures += nbins
                else:
                    self.cat_attributes.append(attr_name)
                    cat_bins = cdict['bins']
                    cat_labels = [f'({cat_bins[i]},{cat_bins[i+1]}]' for i in range(len(cat_bins)-1)]
                    self.category_codes[attr_name] = cat_labels
                    self.nfeatures += len(cat_labels)

            elif dtype == CATEGORICAL or dtype == ORDINAL:
                self.cat_attributes.append(attr_name)
                self.category_codes[attr_name] = cdict['i2s']
                self.nfeatures += len(cdict['i2s'])

        self.__name__ = 'Histogram'

    def extract(self, data):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'

        assert all([c in list(data) for c in self.cat_attributes]), 'Missing some categorical attributes in input data'
        assert all([c in list(data) for c in self.num_attributes]), 'Missing some numerical attributes in input data'

        features = []
        for attr in self.num_attributes:
            col = data[attr]
            F = col.value_counts(bins=self.histogram_bins[attr]).values
            features.extend(F.tolist())

        for attr in self.cat_attributes:
            col = data[attr]
            col = col.astype(CategoricalDtype(categories=self.category_codes[attr], ordered=True))
            F = col.value_counts().loc[self.category_codes[attr]].values
            features.extend(F.tolist())

        assert len(features) == self.nfeatures, f'Expected number of features is {self.nfeatures} but found {len(features)}'

        return array(features)

    def _get_names(self):
        feature_names = []
        for attr in self.num_attributes:
            bins = self.histogram_bins[attr]
            feature_names.extend([f'{attr}({int(bins[i-1])},{int(bins[i])}]' for i in range(1,len(bins))])

        for attr in self.cat_attributes:
            feature_names.extend([f'{attr}_{c}' for c in self.category_codes[attr]])

        return feature_names

