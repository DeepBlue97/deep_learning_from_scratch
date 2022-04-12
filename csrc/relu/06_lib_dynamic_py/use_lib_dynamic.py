import os
import sys

sys.path.append(os.path.join(os.getcwd(), './build'))

import relu

r = relu.Relu()
r.showMask()

print('end')
