#encoding=utf-8
from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET
import logging
import os
import os.path
import time 
import numpy as np
import tensorflow as tf
import random
from PIL import Image
 
# 图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 90
MAX_CAPTCHA = 4
print("Max number of label:", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐
 
# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
 
"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""
 
# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
 
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c =='_':
            k = 62
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map') 
        return k
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
 
"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""

def list_all_file(rootDir,all_file): 
    if not os.path.exists(rootDir) or not os.path.isdir(rootDir):
        return False
    try:
        for lists in os.listdir(rootDir): 
            path = os.path.join(rootDir, lists) 
            all_file.append(path)
            if os.path.isdir(path): 
                list_all_file(path,all_file) 
    except Exception as e:
        print(e)
        return False
    return True

def classify_all_file(all_file):
    data_dict = {}
    for item in all_file:
        base_name = os.path.basename(item)
        pos = base_name.find(".jpg")
        base_chars = base_name[:pos]
        for c in base_chars:
            if c in data_dict.keys():
                file_list = data_dict[c]
                file_list.append(item)
                data_dict[c] = file_list
            else:
                file_list = [item]
                data_dict[c] = file_list
    return data_dict
                

g_train_all_files = []
list_all_file("./train_set",g_train_all_files)
g_test_all_files = []
list_all_file("./test_set",g_test_all_files)
g_train_files_dict = classify_all_file(g_train_all_files)

 
# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
 
    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image(index):
        keys = g_train_files_dict.keys()
        key_index = index % len(keys)
        key = keys[key_index]
        random.seed()
        file_item = random.choice(g_train_files_dict[key])
        base_name = os.path.basename(file_item)
        pos = base_name.find(".jpg") 
        text = base_name[:pos] 
        captcha_image = Image.open(file_item)
        image = np.array(captcha_image)
        if image.shape == (32, 90, 3):
            return text, image
 
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image(i)
        image = convert2gray(image)
 
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
 
    return batch_x, batch_y

def get_next_test_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
 
    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        random.seed()
        file_item = random.choice(g_test_all_files)
        base_name = os.path.basename(file_item)
        pos = base_name.find(".jpg") 
        text = base_name[:pos] 
        captcha_image = Image.open(file_item)
        image = np.array(captcha_image)
        if image.shape == (32, 90, 3):
            return text, image
 
    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
 
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0
        batch_y[i,:] = text2vec(text)
 
    return batch_x, batch_y
 
####################################################################

with tf.name_scope('input'): 
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout
 
# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    #print("******************",IMAGE_HEIGHT,IMAGE_WIDTH)
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
 
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    print("******.conv1",conv1.get_shape())
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    print("******.after conv1",conv1.get_shape())
 
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    print("******.conv2",conv2.get_shape())
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    print("******.after conv2",conv2.get_shape())
 
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    print("******.conv3",conv3.get_shape())
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    print("******.after conv3",conv3.get_shape())
 
    # Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([4*12*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    print("******.w_d",w_d.get_shape())
    print("******.connv3",conv3.get_shape())
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    with tf.name_scope('w_out'):
        w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))

    with tf.name_scope('b_out'):
        b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    #out = tf.nn.softmax(out)
    return out
 
# 训练
def train_crack_captcha_cnn():
    #with tf.device('/cpu:0'):
    output = crack_captcha_cnn()
    # loss
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        tf.summary.scalar('loss',loss) # 可视化loss常量
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
 
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy',accuracy)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/",sess.graph)

        sess.run(tf.global_variables_initializer())
 
        step = 0
        is_save_dot_6 = True
        is_save_dot_7 = True
        is_save_dot_75 = True
        while True:
            batch_x, batch_y = get_next_batch(256)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            #print(step, loss_) 
            logging.debug("step:{%d},loss:{%f}",step,loss_)

            #writer.add_summary(summary,step)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_test_batch(100)
                summary, acc = sess.run([merged, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                #print(step, acc)
                logging.debug("step:{%d},acc:{%f}",step,acc)

                writer.add_summary(summary,step)

                # 如果准确率大于50%,保存模型,完成训练
                #if step == 7000:
                if acc > 0.6 and is_save_dot_6:
                    saver.save(sess, "verify_code.model", global_step=step)
                    is_save_dot_6 = False
                if acc > 0.7 and is_save_dot_7:
                    saver.save(sess, "verify_code.model", global_step=step)
                    is_save_dot_7 = False
                if acc > 0.75 and is_save_dot_75:
                    saver.save(sess, "verify_code.model", global_step=step)
                    is_save_dot_75 = False
                if acc > 0.8:
                    saver.save(sess, "verify_code.model", global_step=step)
                    break
 
            step += 1
            #performance test
            #if step == 20:
            #    break

if __name__ == '__main__': 
    start = time.clock()
    LOG_FORMAT = '%(asctime)s-%(levelname)s-[%(process)d]-[%(thread)d] %(message)s (%(filename)s:%(lineno)d)'
    logging.basicConfig(format=LOG_FORMAT,level=logging.DEBUG,filename="./ts.log",filemode='w')

    train_crack_captcha_cnn()

    end = time.clock()
    print('Running time: %s Seconds'%(end - start))
