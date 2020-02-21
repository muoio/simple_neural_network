A = 0.35
B = 0.9
w1 = 0.3
w2 = 0.9
w3 = 0.1
w4 = 0.8
w5 = 0.4
w6 = 0.6
err_final = 1
learning_rate = 0.00003
s = [(92,28,40.0),(82,64,48.7),(38,88,42.0),(18,36,18.0),(30,7,12.3),(39,21,20.0),(59,16,25.0),(24,18,14.0),(5,43,16.0),(10,84,31.3),(55,22,25.7),(98,35,44.3),(86,38,41.3),(31,62,31.0),(37,48,28.3),(90,74,54.7),(74,40,38.0),(49,70,39.7),(92,57,49.7),(25,18,14.3),(92,42,44.7),(27,45,24.0),(17,77,31.3),(22,11,11.0),(35,52,29.0),(3,5,2.7),(20,72,30.7),(89,39,42.7),(38,78,38.7),(38,18,18.7)]
cnt = 0
err = []
import math
def elu(x, diff = 0):#elu activation function
	if not diff:
		if (x>=0):
			return x
		else: 
			return 0.01*(math.exp(x)-1)
	else:
		if(x>0): 
			return 1
		else:
			return 0.01*math.exp(x)

while cnt <5000:#train the neural network
	sum_err = 0
	for A,B,target in s:
		top = A*w3 + B*w4
		bottom = A*w5 + B*w6
		final = top*w1 + bottom*w2
		err_final = (target-final)*elu(final,diff=1) # caculate error of output node
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
A = 60 
B = 90 
top = A*w3 + B*w4
bottom = A*w5 + B*w6
final = top*w1 + bottom*w2
print final
import matplotlib.pyplot as plt
plt.plot(err)
plt.show() # show error value in graph
