class test():
    def __init__(self):
        self.x = 200
    def __call__(self, y):
        print('called ', y)



import tensorflow as tf
D = 3
print(tf.matrix_band_part(tf.ones([D, D], -1, 0)))
