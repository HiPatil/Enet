from Enet import *
from data_processing import *
from keras.callbacks import TensorBoard
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


data_gen_args = dict()
tensorboard = TensorBoard(log_dir='./logs_multi_class_iou')

car = trainGenerator(8,'/home/himanshu/dl/number_plate_detection/Enet/dataset/validation','images','masks',data_gen_args,image_color_mode = "rgb", flag_multi_class = True, num_class=8, target_size = (512,512), save_to_dir = None)
model = Enet(num_classes = 8)
model_checkpoint = ModelCheckpoint('/home/himanshu/dl/number_plate_detection/Enet/ENET_multiclass_iou.hdf5', monitor = 'loss', verbose=1, save_best_only=True)
model.fit_generator(car, steps_per_epoch = 250, epochs = 100, callbacks = [tensorboard,model_checkpoint])
