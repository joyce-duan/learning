from sklearn.base import BaseEstimator,  TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

class TextStats(BaseEstimator, TransformerMixin):
	"""Extract features from each document for DictVectorizer"""
	def __init__(self):
		self.dict_vectorizer = DictVectorizer(sparse=False)

	def fit(self, x, y=None):
		#self.dict_vectorizer.fit(d)
		return self

	def transform(self, posts):
		d  = [{'length': len(text),
				'num_sentences': text.count('.')
				,'num_words': len(text.split()) }
				for text in posts]
		return self.dict_vectorizer.fit_transform(d)

	def get_feature_names(self):
		return self.dict_vectorizer.get_feature_names()

class SpecialWordCounter(BaseEstimator):
    def __init__(self, fname = '../data/exclude_words.txt', prefix = 'test'):
        with open(fname) as f:
            badwords = [l.strip() for l in f.readlines()]
        self.badwords_ = [w for w in badwords if len(w) >=2]
        self.prefix = prefix

    def get_feature_names(self):
        return np.array(['ratio_' + self.prefix, 'n_'+self.prefix])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        ## some handcrafted features!
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]

        '''
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                                            for c in documents]

        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        '''
        # number of special words:
        n_special = [np.sum([c.lower().count(w) for w in self.badwords_])
                                                for c in documents]
        special_ratio = np.array(n_special) / np.array(n_words, dtype=np.float)
        print n_words
        print n_special
        print self.badwords_

        return np.array([special_ratio, n_special,
            ]).T

if __name__ == '__main__':
	t = ['this is a gps therapy test.'
		, 'a second one. hope this is better.'
		, 'more dancing club night.']
	text_stats = TextStats()
	text_stats.fit(t)
	print t
	print 'text stats'
	x = text_stats.transform(t)	
	print text_stats.get_feature_names()	
	print x

	print ''
	word_count = SpecialWordCounter()
	word_count.fit(t)
	print word_count.get_feature_names()
	print word_count.transform(t)

	combined_features = FeatureUnion([
		('stats', TextStats())
		, ('special_word_stats', SpecialWordCounter())
		])

	# Use combined features to transform dataset:
	X_features = combined_features.fit(t).transform(t)
	print '\nfeature union'
	print 'X:', X_features
	print 'names:', combined_features.get_feature_names()
	print 

	pipeline = Pipeline([
	    # Use FeatureUnion to combine the features from subject and body
	    ('union', FeatureUnion(
	        transformer_list=[
	        ('scaled_text_stats', Pipeline([
                ('stats', TextStats())
               , ('scaling',  StandardScaler())
            ])
            )
	        , ('special_word_stats', SpecialWordCounter())
	        ]
	        )
	    )
	    ])
	pipeline.fit(t)
	x_all = pipeline.transform(t)
	print '\npipeline to union features'
	print x_all
	#print pipeline.named_steps['union'].get_feature_names()
