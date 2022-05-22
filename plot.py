import matplotlib.pyplot as plt
import numpy as np

with open('position_time_data.txt') as f:
    lines = f.readlines()[1:]
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]

x_floats, y_floats,predicted_positions = [],[],[]

for x1 in x:
    x_floats.append(float(x1))
for y1 in y:
    y_floats.append(float(y1))

reg_coeffs,residuals,_,_,_=np.polyfit(y_floats,x_floats,1,full=True)
A,B=reg_coeffs
print(A,B)
time_array=np.array(y_floats)
y_regression = A*time_array+B

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.axes.xaxis.set_visible(False)
# ax1.axes.yaxis.set_visible(False)
# ax1.scatter(y,x)
# ax1.plot(y,predicted_positions)


# plt.show()