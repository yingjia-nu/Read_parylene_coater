import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np



# Create a vector, scalar, matrix and tensor with values of your choosing using tf.constant().
# Find the shape, rank and size of the tensors you created in 1.
scalar = tf.constant(10)
print(f"scalar's shape: {scalar.shape}")
print(f"scalar rank: {scalar.ndim}")
print(f"scalar size: {tf.size(scalar)}")

vector = tf.constant([0,10])
print(vector)
print(f"vector's shape: {vector.shape}")
print(f"vector rank: {vector.ndim}")
print(f"vector size: {tf.size(vector)}")

matrix = tf.constant([[1, 2],[3,4]])
print(f"matrix is {matrix}")
print(f"matrix's shape: {matrix.shape}")
print(f"matrix rank: {matrix.ndim}")
print(f"matrix size: {tf.size(matrix)}")

tensor = tf.constant([[[1, 2],[3,4]],[[1, 2],[3,4]]])
print(tensor)
print(f"tensor's shape: {tensor.shape}")
print(f"tensor rank: {tensor.ndim}")
print(f"tensor size: {tf.size(tensor)}")

# Create two tensors containing random values between 0 and 1 with shape [5, 300]
random_1 = tf.random.Generator.from_seed(11)
random_1 = random_1.uniform(shape=(5,300))

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.uniform(shape=(5,300))
# Multiply the two tensors you created in 3 using matrix multiplication.
mul = tf.matmul(random_1, tf.transpose(random_2))
mul_reshape = tf.matmul(random_1, tf.reshape(random_2, shape = (300, 5)))
print(mul_reshape)

# Multiply the two tensors you created in 3 using dot product.
dot_1 = tf.tensordot(tf.transpose(random_1), random_2, axes=1)
print(dot_1.shape)
# Create a tensor with random values between 0 and 1 with shape [224, 224, 3].
random_3 = tf.random.Generator.from_seed(12)
random_3 = random_3.uniform(shape=(224, 224, 3))
print(random_3.shape)
# Find the min and max values of the tensor you created in 6.
print(f"min of random tensor: {tf.reduce_min(random_3)}")
print(f"max of random tensor: {tf.reduce_max(random_3)}")


# Created a tensor with random values of shape [1, 224, 224, 3] then squeeze it to change the shape to [224, 224, 3]
random_4 = tf.random.Generator.from_seed(14)
random_4 = random_4.normal(shape=(1, 224, 224, 3))
random_4s = tf.squeeze(random_4)
print(f"squeezed random_4, shape: {random_4s.shape}")

# Create a tensor with shape [10] using your own choice of values, then find the index which has the maximum value.
tensor_1 = tf.constant(np.arange(10), shape=(10,))
print(f"index of the max value is {tf.argmax(tensor_1).numpy()}")

# One-hot encode the tensor you created in 9.
one_hot = tf.one_hot(np.arange(6), depth=8)
print(one_hot)