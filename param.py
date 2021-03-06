import matplotlib as mpl
import spectral as spy
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
spy.settings.WX_GL_DEPTH_SIZE = 100

CLASSES_NUM = 6  # 输出7类地物
LABELS = ['', 'Trees', 'Asphalt', 'Parking lot', 'Bitumen', 'Meadow', 'Soil']
VAL_FRAC = 0.5
TEST_FRAC = 0.3  # target用来测试数据的百分比 test/train
TRAIN_FRAC = 0.7
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
PATIENCE = 15
BUFFER_SIZE = 60000
BATCH_SIZE = 50
BANDS = 72
EPOCHS = 1000
noise_dim = 72
num_examples_to_generate = 16
seed = tf.random.normal([BATCH_SIZE, 72, 1])
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
FEATURE_dim = 36
lr = 4e-4
