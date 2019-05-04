from sklearn.base import TransformerMixin
import numpy as np

class DFTTransformer(TransformerMixin):
    def __init__(self, sample_rate=16000, feature_num=400, aggregator=sum):
        assert sample_rate % feature_num == 0, "Feature_num must be a divisor of the sample rate"
        self.sample_rate_ = sample_rate
        self.feature_num_ = feature_num
        self.aggregator_ = aggregator

    def fit_transform(self, X, y=None, **fit_params):
        N = X.shape[1]
        assert N % self.sample_rate_ == 0, "The number of features must be an integer multiple of the sampling rate"

        features = np.zeros((0, self.feature_num_))
        bucket_size = int(N / self.feature_num_)

        dft = np.abs(np.fft.fft(X, axis=1) / N)

        for i in range(len(dft)):
            data_vec = dft[i, :]
            feature_vec = np.array([self.aggregator_(data_vec[x:x+bucket_size]) for x in range(0, len(data_vec), bucket_size)])
            features = np.vstack((features, feature_vec))

        return features

