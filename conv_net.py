import numpy as np
import data_helpers as dh

import matplotlib.pyplot as plt

import tensorflow as tf
import datetime as dt




#X,Y,files = dh.load_dataset('shared/Digits_2',(56,32),2)



# import numpy.random as rnd
# for a in xrange(10):
    
#     k = rnd.randint(Y.shape[0])
#     im= X[k,:].reshape((32,56))
#     plt.imshow(im,cmap=plt.cm.gray, interpolation='bicubic')
#     print("file="+files[k], Y[k,:].reshape((2,10)) )
#     plt.show()
    
    


# Parameters
learning_rate = 0.001
batch_size = 64
training_iters =100*batch_size 
display_step = 10

# Network Parameters
n_input = 32*56 
n_classes = 2*10 # each word is encoded by 1260 vector
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 32, 56, 1])

    # Convolution Layer 3x3x32 first, layer with relu
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling), change input size by factor of 2 
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)
    
    # Convolution Layer, 3x3x64
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    #out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])), # 3x3 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])), # 3x3 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([8*14*64, 1024])), # fully connected, 64/(2*2*2)=8, 304/(2*2*2)=38 (three max pool k=2) inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, 2*10])) # 1024 inputs, 20*63 outputs for one catpcha word (max 20chars)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([2*10]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer


############
# splited softmax_cross_entropy loss
#split prediction for each char it takes 63 continous postions, we have 20 chars
# split_pred = tf.split(1,20,pred)
# split_y = tf.split(1,20,y)


# #compute partial softmax cost, for each char
# costs = list()
# for i in range(20):
#     costs.append(tf.nn.softmax_cross_entropy_with_logits(split_pred[i],split_y[i]))
    
# #reduce cost for each char
# rcosts = list()
# for i in range(20):
#     rcosts.append(tf.reduce_mean(costs[i]))
    
# # global reduce    
# loss = tf.reduce_sum(rcosts)



cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(pred,y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Evaluate model

# pred are in format batch_size,20*63, reshape it in order to have each character prediction
# in row, then take argmax of each row (across columns) then check if it is equal 
# original label max indexes
# then sum all good results and compute mean (accuracy)

#batch, rows, cols
p = tf.reshape(pred,[batch_size,2,10])
#max idx acros the rows
#max_idx_p=tf.argmax(p,2).eval()
max_idx_p=tf.argmax(p,2)

l = tf.reshape(y,[batch_size,2,10])
#max idx acros the rows
#max_idx_l=tf.argmax(l,2).eval()
max_idx_l=tf.argmax(l,2)

correct_pred = tf.equal(max_idx_p,max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))







#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()


losses = list()
accuracies = list()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    epoch=0
    start_epoch=dt.datetime.now()
    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys, idx = dh.random_batch(X, Y, batch_size)
        
        
        # Fit training using batch data
        print("##############")
        print("opt step {}".format(dt.datetime.now()))
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        print("end step {}".format(dt.datetime.now()))        
        
        if step % display_step == 0:
            
            print("acc start {}".format(dt.datetime.now()))
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            accuracies.append(acc)
            
            print("loss start {}".format(dt.datetime.now()))
            # Calculate batch loss
            batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            losses.append(batch_loss)
            
            print("Iter " + str(step*batch_size) + " started={}".format(dt.datetime.now()) + ", Minibatch Loss= " + "{}".format(batch_loss) + ", Training Accuracy= " + "{}".format(acc))
            
            batch_idx=0
            k=idx[batch_idx]
            
            pp = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            p = tf.reshape(pp,[batch_size,2,10])
            max_idx_p=tf.argmax(p,2).eval()
            
            predicted_word = vecmp.map_vec_pos2words(max_idx_p[batch_idx,:])


            l = tf.reshape(batch_ys,[batch_size,2,10])
            #max idx acros the rows
            max_idx_l=tf.argmax(l,2).eval()
            true_word = vecmp.map_vec_pos2words(max_idx_l[batch_idx,:])

            
            print("true : {}, predicted {}".format(true_word, predicted_word))

            
            
            epoch+=1
        
        step += 1
        
        if step%1000==0:
            save_path = saver.save(sess, "./model_sigmoid.ckpt")
        
        
    end_epoch = dt.datetime.now()
    print("Optimization Finished, end={} duration={}".format(end_epoch,end_epoch-start_epoch))
    
    
    test_size = min(1000, X.shape[0])
    test_X = X[0:test_size,:]
    test_Y = Y[0:test_size,:]
    # Calculate accuracy 
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}))
    

import matplotlib.pyplot as plt

# plt.plot(losses)
# plt.plot(accuracies)

plt.figure(1)
plt.subplot(211)
plt.plot(losses, '-b', label='Loss')
plt.title('Loss function')
plt.subplot(212)
plt.plot(accuracies, '-r', label='Acc')
plt.title('Accuracy')
    
    