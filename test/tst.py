import numpy as np

summation = 1

output = 0 if summation < -144 else 1 / (1 + np.power(np.e, 4.9 * -summation))
print(output)

output = 0 if summation < -709 else 1/(1 + np.power(np.e, -summation))
print(output)
