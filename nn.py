#init values for the network
A = 0.35
B = 0.9
w1 = 0.3
w2 = 0.9
w3 = 0.1
w4 = 0.8
w5 = 0.4
w6 = 0.6
# see the init_network.png for detail
err_final = 1
learning_rate = 0.01
s = [(0.2,0.5,0.23),(0.3,0.9,0.4),(0.5,0.5,0.33),(0.1,0.9,0.33)] # train data set
cnt = 0
err = []
while cnt <10000: #train the network
	sum_err = 0
	for A,B,target in s:
		top = A*w3 + B*w4
		bottom = A*w5 + B*w6
		final = top*w1 + bottom*w2
		err_final = (target-final)*(1-final)*final
		sum_err += err_final
		w1 += err_final*top*learning_rate
		w2 += err_final*bottom*learning_rate
		err_top = err_final*w1
		err_bottom = err_final*w2
		w3 += err_top * A *learning_rate
		w4 += err_top * B *learning_rate
		w5 += err_bottom * A * learning_rate
		w6 += err_bottom * B * learning_rate
		cnt+=1
	err.append(err_final)

A = 0.6
B = 0.9 # A,B input which you want to predict
top = A*w3 + B*w4
bottom = A*w5 + B*w6
final = top*w1 + bottom*w2 # prediction
print final
print w1, w2, w3, w4, w5, w6
print cnt
import matplotlib.pyplot as plt
plt.plot(err) 
plt.show()# show loss function value
