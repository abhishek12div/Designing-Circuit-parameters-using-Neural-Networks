import tensorflow as tf
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=14)
SMALL_SIZE = 10
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def normalize(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom, X.min(axis=0), X.max(axis=0)

def denormalize(X, x_min, x_max):
    return X*(x_max-x_min)+x_min


#H:\Research\NN Designer of an Amplifier\New data\Amp csv files
from numpy import genfromtxt

#Technical Presentation data
Data = genfromtxt("..........csv",delimiter=',')
Label = genfromtxt("..........csv",delimiter=',')


X_data, min_x , max_x = normalize(Data, 0, 1)

Y_data, min_y, max_y = normalize(Label, 0, 1)


from sklearn.model_selection import train_test_split
X, X1, Y, Y1 = train_test_split(X_data, Y_data, test_size=0.3, shuffle=True)



print(tf.__version__)
print(X.shape)
print(Y.shape)

input_nodes = X.shape[1]
h1 = 5
h2 = 18
#h3 = 8
output_nodes = Y.shape[1]

#seed = np.random.randint(0,100)
tf.set_random_seed(68)
#print(seed)

data = tf.placeholder(dtype=tf.float32, shape=[None,input_nodes])
label = tf.placeholder(dtype=tf.float32, shape=[None, output_nodes])



#input to 1st hidden
fully_connected1 = tf.contrib.layers.fully_connected(inputs=data,
                                                     num_outputs=h1,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5,0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5,0.5)
                                                     )

#1st hidden to 2nd
fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1,
                                                     num_outputs=h2,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                                     )

#2nd to 3rd
'''fully_connected3 = tf.contrib.layers.fully_connected(inputs=fully_connected2,
                                                     num_outputs=h3,
                                                     activation_fn=tf.nn.sigmoid,
                                                     weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                                     biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                                     )'''

#3rd hidden to ouput
pred = tf.contrib.layers.fully_connected(inputs=fully_connected2,
                                         num_outputs=output_nodes,
                                         activation_fn=tf.nn.sigmoid,
                                         weights_initializer=tf.random_uniform_initializer(-0.5, 0.5),
                                         biases_initializer=tf.random_uniform_initializer(-0.5, 0.5)
                                         )

error_func = 0.5 * tf.reduce_mean(tf.square(pred - label))

model_sgd = tf.train.GradientDescentOptimizer(0.01).minimize(error_func)
model_mom = tf.train.MomentumOptimizer(0.01,momentum=0.95,use_nesterov=False).minimize(error_func)
model_nag = tf.train.MomentumOptimizer(0.01,momentum=0.9,use_nesterov=True).minimize(error_func)
model_rms = tf.train.RMSPropOptimizer(0.001).minimize(error_func)
model_adam = tf.train.AdamOptimizer(0.001).minimize(error_func)
#model_adagrad = tf.train.AdagradOptimizer(0.001).minimize(error_func)

max_iters = 10001
error_bfgs = deque(maxlen=max_iters)
model_bfgs = tf.contrib.opt.ScipyOptimizerInterface(error_func, options={'disp':True, 'maxiter':max_iters,
                                                                         'error_list':error_bfgs, 'gtol':1e-5}, method='bfgs')
error_naq = deque(maxlen=max_iters)
model_naq = tf.contrib.opt.ScipyOptimizerInterface(error_func, options={'disp':True, 'maxiter':max_iters,
                                                                         'error_list':error_naq,'gtol':1e-6, 'u':0.85}, method='naq')

init = tf.global_variables_initializer()

algo = ['GD', 'CM', 'NAG', 'RMS', 'Adam', 'QN', 'NAQ']
test_error = []
for meth in algo:
    with tf.Session() as sess:
        sess.run(init)
        if meth == 'GD':
            cf_sgd = []
            for i in range(max_iters):
                result,err_sgd = sess.run([model_sgd,error_func], feed_dict={data:X, label:Y})
                if i % 1000 == 0:
                    print("Cost after iteration %i %f" %(i, err_sgd))
                cf_sgd.append(err_sgd)
            print('Train error(SGD) is {}'.format(err_sgd))
                
            test_err = sess.run(error_func, feed_dict={data:X1, label:Y1})
            test_error.append(test_err)
            print('Test error is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted outputs:\n',denormalize(nn_out, min_y, max_y))


        if meth == 'CM':
            cf_mom = []
            for i in range(max_iters):
                result1, err_mom = sess.run([model_mom, error_func], feed_dict={data:X, label:Y})
                if i % 1000 == 0:
                    print("Cost after iteration %i %f" %(i, err_mom))
                cf_mom.append(err_mom)
            print('Train error(MOM) is {}'.format(err_mom))
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output:\n',denormalize(nn_out, min_y, max_y))

        if meth == 'NAG':
            cf_nag = []
            for i in range(max_iters):
                result1, err_nag = sess.run([model_nag, error_func], feed_dict={data:X, label:Y})
                if i % 1000 == 0:
                    print("Cost after iteration %i %f" %(i, err_nag))
                cf_nag.append(err_nag)
            print('Train error(NAG) is {}'.format(err_nag))
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output:\n',denormalize(nn_out, min_y, max_y))

        if meth == 'RMS':
            cf_rms = []
            for i in range(max_iters):
                result3, err_rms = sess.run([model_rms, error_func], feed_dict={data: X, label: Y})
                if i % 1000 == 0:
                    print("Cost after iteration %i %f" %(i, err_rms))
                cf_rms.append(err_rms)
            print('Train error(RMS) is {}'.format(err_rms))
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output\n',denormalize(nn_out, min_y, max_y))

        if meth == 'Adam':
            cf_adam = []
            cf_err_test = []
            for i in range(max_iters):
                result4, err_adam = sess.run([model_adam, error_func], feed_dict={data: X, label: Y})
                if i % 1000 == 0:
                    print("Cost after iteration %i %f" %(i, err_adam))
                cf_adam.append(err_adam)
                err_test_sgd = sess.run(error_func, feed_dict={data: X1, label: Y1})
                if i % 1000 == 0:
                    print("Test Cost %f" % (err_test_sgd))
                cf_err_test.append(err_test_sgd)
            print('Train error(ADAM) is {}'.format(err_adam))
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output:\n',denormalize(nn_out, min_y, max_y))

        if meth == 'QN':
            res_bfgs = model_bfgs.minimize(sess, feed_dict={data: X, label: Y})
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error(QN) is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output:\n',denormalize(nn_out, min_y, max_y))


        if meth == 'NAQ':
            res_naq = model_naq.minimize(sess, feed_dict={data: X, label: Y})
            test_err = sess.run(error_func, feed_dict={data: X1, label: Y1})
            test_error.append(test_err)
            print('Test error(NAQ) is {}\n'.format(test_err))
            denorm_y = denormalize(Y1[:1], min_y, max_y)
            print('actual outputs:\n', denorm_y)
            nn_out = sess.run(pred, feed_dict={data: X1[:1]})
            print('predicted output:\n',denormalize(nn_out, min_y, max_y))


plt.semilogy(cf_sgd, label='GD', color='black', linestyle='dotted')

plt.semilogy(cf_mom, label='CM(μ=0.95)', color='black', linestyle='solid')
plt.semilogy(cf_nag, label='NAG(μ=0.9)', color='black', linestyle='dashed')
plt.semilogy(cf_rms, label='RMSprop', color='black', linestyle='dashdot')
plt.semilogy(cf_adam, label='Adam', color='black', linestyle='dotted')
#plt.semilogy(cf_err_test, label='Testerr_GD')
plt.semilogy(error_bfgs,label='QN', color='grey', linestyle='solid')
plt.semilogy(error_naq,label='NAQ(μ=0.85)', color='black', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(loc='best')
plt.tight_layout()

print(test_error)

plt.show()



'''plt.figure()
plt.xticks(range(len(algo)), algo, rotation=0)
plt.bar(range(len(algo)),test_error, 0.2)
plt.show()'''
