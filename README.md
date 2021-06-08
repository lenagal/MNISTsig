# MNISTsig
Applies signature method to learn sequentialised MNIST digits.
The MNIST data is transformed into sequences of pixels using https://edwin-de-jong.github.io/blog/mnist-sequence-data/
We then use [esig](https://github.com/datasig-ac-uk/esig/blob/develop/README.md) to vectorise the data by taking signatures of paths representing the digits. 
The objective is to conduct experiments with signature methosd and various feature selection on this dataset.
