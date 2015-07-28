import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import ml_util



def test_count_null():
	l1 = [1, np.nan,2,3]
	l2 = [np.nan,'b',np.nan, 'c']
	l3 = [1, 2, 3, 4]
	print l1, l2, l3
	print len(l1), len(l2)
	df = pd.DataFrame({'c1':l1,'c2':l2, 'c3':l3})
	print ml_util.count_null(df)


if __name__ == '__main__':
	test_count_null()