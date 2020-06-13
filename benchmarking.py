import time
import numpy as np


# start = time.time()
#
# a = 0
# for i in range(999999):
#     a = np.random.normal(0, 1)
#
# end = time.time()
#
# print(end - start)


start = time.time()

print(np.random.normal(0, 1, (999999,)).shape)

end = time.time()

print(end - start)
