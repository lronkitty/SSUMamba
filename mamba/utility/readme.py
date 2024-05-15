from read_HSI import read_HSI
from refold import refold
kernel_size = (x,y,z)
stride =(x_,y_,z_)
col_data,data_shape = read_HSI('''cube''',kernel_size=kernel_size,stride=stride)
#col_data.shape = [n,1,x,y,z] 对col_data进行各种运算，保证shape不变
'''cube''' = refold(col_data,data_shape=data_shape, kernel_size=kernel_size,stride=stride)
