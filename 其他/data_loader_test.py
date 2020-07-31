import numpy as np
import paddle.fluid as fluid
BATCH_NUM = 10
BATCH_SIZE = 16
MNIST_IMAGE_SIZE = 784
MNIST_LABLE_SIZE = 1

# 伪数据生成函数，服务于下述三种不同的生成器
def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label

# 每次生成一个Sample，使用set_sample_generator配置数据源
def sample_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM * BATCH_SIZE):
            image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
            yield image, label

    return __reader__

# 每次生成一个Sample List，使用set_sample_list_generator配置数据源
def sample_list_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            sample_list = []
            for _ in range(BATCH_SIZE):
                image, label = get_random_images_and_labels([MNIST_IMAGE_SIZE], [MNIST_LABLE_SIZE])
                sample_list.append([image, label])

            yield sample_list

    return __reader__

# 每次生成一个Batch，使用set_batch_generator配置数据源
def batch_generator_creator():
    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, MNIST_IMAGE_SIZE], [BATCH_SIZE, MNIST_LABLE_SIZE])
            yield batch_image, batch_label

    return __reader__


BATCH_SIZE = 10

place = fluid.CPUPlace() # 或者 fluid.CUDAPlace(0)
fluid.enable_imperative(place)
#
# # 使用sample数据生成器作为DataLoader的数据源
# data_loader1 = fluid.io.DataLoader.from_generator(capacity=10)
# data_loader1.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, places=place)
#
# # 使用sample list数据生成器作为DataLoader的数据源
# data_loader2 = fluid.io.DataLoader.from_generator(capacity=10)
# data_loader2.set_sample_list_generator(sample_list_generator_creator(), places=place)

# 使用batch数据生成器作为DataLoader的数据源
data_loader3 = fluid.io.DataLoader.from_generator(capacity=10)
data_loader3.set_batch_generator(batch_generator_creator(), places=place)


for data in data_loader3():
    image, label = data
