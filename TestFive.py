from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()
print(digits.data.shape)

pl.gray()
pl.matshow(digits.images[0])
pl.show()
