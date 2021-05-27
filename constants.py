# import keras
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adadelta 

IMG_WIDTH, IMG_HEIGHT = 256, 256
NB_TRAIN_SAMPLES = 5168
NB_VALIDATION_SAMPLES = 64
EPOCHS = 1
BATCH_SIZE = 16
NUM_CLASSES = 2

TENSORBOARD_DIR = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/logs'
CHECKPOINT_PATH = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/resnetmodel.hdf5'
CHECKPOINT_PATH1 = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/vggmodel.hdf5'
WEIGHTS_PATH1 = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/resnetweights.h5'
WEIGHTS_PATH2 = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/vggweights.h5'
MODELPATH = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/resnetmodel.hdf5'
MODELPATH1 = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray_code/vggmodel.hdf5'
DATA_PATH = 'C:/Users/AG92031/OneDrive - Anthem/Desktop/WorkOS/Research Work/chest_xray/' 
TRAIN_DATA = DATA_PATH+ 'train/'
TEST_DATA = DATA_PATH+ 'test/'
VAL_DATA = DATA_PATH+ 'val/' 

CHECKPOINTER = ModelCheckpoint(monitor='categorical_accuracy',
    filepath=CHECKPOINT_PATH, verbose=1, save_best_only=True)

TENSORBOARD = TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=0,
                          write_graph=True, write_images=False)

ADADELTA = Adadelta(lr = 0.01, rho = 0.00001)

SGD = SGD(lr=0.01, decay = 1e-6, momentum = 0.98, nesterov = True)

LR_REDUCER = ReduceLROnPlateau(patience = 5, monitor = 'loss', factor = 0.95, verbose = 1)