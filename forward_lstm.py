import numpy as np
import util

x_test, y_test = util.get_data("test")

sample_size = 4
sample_shape = (2, 2)

sample_index = np.arange(sample_size)

x = x_test[sample_index]
y = y_test[sample_index]

print x, y