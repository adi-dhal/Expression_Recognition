import tensorflow as tf
import numpy as  np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
clf = SVC(kernel='linear', probability=True, tol=1e-3)

n_nodes_hl1 = 300
n_nodes_hl2 = 300
n_nodes_hl3 = 300

n_classes = 4

batch_size = 10

train_data = []
train_lab = []
test_data = []
test_lab = []

x = tf.placeholder('float', [None, 139])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([139, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	
	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	#output = tf.nn.relu(output)
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	hm_epochs = 50
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_data):
				start = i
				end =i+batch_size
				batch_x =np.array(train_data[start:end])
				batch_y =np.array(train_lab[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i = i + batch_size
		saver.save(sess,"model.ckpt")
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:np.array(test_data), y:np.array(test_lab)}))		
		
		
def extract_data(flag):
	global train_data,train_lab ,test_data ,test_lab
	with open("feature_vector.txt",'r') as fl:
		for line in fl:
			line = line.split('\n')[0]
			data,lab = line.split('_')
			data = data.split(',')
			lab = lab.split(',')
			train_data.append(map(float,data))
			if flag == 1:
				train_lab.append(map(float,lab)) # for neural network
			elif flag == 2: 
				train_lab.append(np.argmax(lab))	# for SVM
	with open("feature_vector_test.txt","r") as fl:
		for line in fl:
			line = line.split('\n')[0]
			data,lab = line.split('_')
			data = data.split(',')
			lab = lab.split(',')
			test_data.append(map(float,data))
			if flag == 1:
				test_lab.append(map(float,lab))	# for neural network
			elif flag == 2:
				test_lab.append(np.argmax(lab))		# for SVM
def trainSVM():
	extract_data(2)
	clf.fit(np.array(train_data), np.array(train_lab))
	print "Accuracy"
	print clf.score(np.array(test_data), np.array(test_lab))
	print "Confusion_Matrix"
	print confusion_matrix(np.array(test_lab),clf.predict(np.array(test_data)))
def trainNN():
	extract_data(1)
	train_neural_network(x)	
def use_SVM(sample):
	extract_data(2)
	clf.fit(np.array(train_data), np.array(train_lab))
	print clf.predict(np.array(sample).reshape(1,-1))
def use_NN(sample):
	prediction = neural_network_model(x)
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.initialize_all_variables())
		saver.restore(sess,"model.ckpt")
		result = sess.run(prediction,{x:np.array(sample).reshape(1,-1)})
		result = np.argmax(result)		
		print result

	
