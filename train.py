# from os import listdir
# import cv2
# import numpy as np
# import tensorflow as tf
# import pickle
# from tensorflow import keras 
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from keras.api.applications.vgg16 import VGG16
# from keras.api.layers import Input, Flatten, Dense, Dropout
# from keras.api.models import Model
# from keras.api.callbacks import ModelCheckpoint
# from keras.src.callbacks import ModelCheckpoint
# import matplotlib.pyplot as plt
# import random
# import numpy as np
# import skimage.io as io
# from skimage.transform import rotate, AffineTransform, warp
# from skimage.util import random_noise
# from skimage.filters import gaussian
# from sklearn.preprocessing import LabelBinarizer
# from keras.api.callbacks import ModelCheckpoint, EarlyStopping
# from sklearn.utils.validation import check_is_fitted
# raw_folder = "D:/Hocmay/test/data/"

# def apply_augmentation(image):
#     # Định nghĩa các phép biến đổi hình ảnh
#     sigma = 0.155
#     rotated = rotate(image, angle=np.random.uniform(-20, 20), mode='wrap')
#     transform = AffineTransform(translation=(np.random.randint(-25, 25), np.random.randint(-25, 25)))
#     shifted = warp(image, transform, mode='wrap')
#     flipped_horizontal = np.fliplr(image)
#     noisy = random_noise(image, var=sigma**2)
    
#     # Trả về ảnh gốc và các phiên bản tăng cường
#     return rotated, shifted, flipped_horizontal, noisy

# # Hàm lưu dữ liệu
# def save_data(raw_folder="D:/Hocmay/test/data/"):
#     dest_size = (128, 128)
#     print("Bắt đầu xử lý ảnh...")
#     pixels = []
#     labels = []

#     for folder in listdir(raw_folder):
#         if folder != '.DS_Store':
#             print("Folder=", folder)
#             for file in listdir(raw_folder + folder):
#                 if file != '.DS_Store':
#                     print("File=", file)
#                     pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=(128, 128)))
#                     labels.append(folder)

#     pixels = np.array(pixels)
#     labels = np.array(labels)

#     encoder = LabelBinarizer()
#     labels = encoder.fit_transform(labels)

#     file = open('D:/Hocmay/test/pix.data', 'wb')
#     pickle.dump((pixels, labels), file)
#     file.close()

# # Hàm load dữ liệu
# def load_data():
#     file = open('D:/Hocmay/test/pix.data', 'rb')
#     (pixels, labels) = pickle.load(file)
#     file.close()
#     return pixels, labels

# # Tải dữ liệu hoặc tạo mới nếu chưa có
# save_data()
# X, y = load_data()
# random.shuffle(X)

# # Chia dữ liệu thành tập huấn luyện và kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# X_train_augmented = []
# y_train_augmented = []

# # Áp dụng augmentation cho từng ảnh trong tập dữ liệu huấn luyện
# for image, label in zip(X_train, y_train):
#     augmented_images = apply_augmentation(image)
#     for augmented_image in augmented_images:
#         X_train_augmented.append(augmented_image)
#         y_train_augmented.append(label)

# # Chuyển danh sách thành mảng numpy
# X_train_augmented = np.array(X_train_augmented)
# y_train_augmented = np.array(y_train_augmented)
# # Tạo model
# def get_model():
#     model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
#     for layer in model_vgg16_conv.layers:
#         layer.trainable = False

#     input = Input(shape=(128, 128, 3), name='image_input')
#     output_vgg16_conv = model_vgg16_conv(input)

#     x = Flatten(name='flatten')(output_vgg16_conv)
#     x = Dense(512, activation='relu', name='fc1')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256, activation='relu', name='fc2')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(7, activation='softmax', name='predictions')(x)

#     my_model = Model(inputs=input, outputs=x)
#     my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return my_model

# vggmodel = get_model()

# # Định nghĩa callbacks
# filepath = "D:/Hocmay/test/weights-{epoch:02d}-{val_accuracy:.2f}.keras"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]

# # Huấn luyện model
# vgghist = vggmodel.fit(
#     X_train_augmented, y_train_augmented,
#     batch_size=64,
#     epochs=50,
#     validation_data=(X_test, y_test),
#     callbacks=callbacks_list
# )

# vggmodel.save("D:/Hocmay/test/vggmodel.h5")

from os import listdir
import cv2
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.api.applications.vgg16 import VGG16
from keras.api.layers import Input, Flatten, Dense, Dropout
from keras.api.models import Model
from keras.api.callbacks import ModelCheckpoint
from keras.src.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
import skimage.io as io
from sklearn.preprocessing import LabelBinarizer
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.validation import check_is_fitted
import torch
from torchvision import transforms
from PIL import Image

raw_folder = "D:/Hocmay/test/data/"

# Định nghĩa các phép biến đổi và tăng cường dữ liệu
transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Hàm áp dụng augmentation sử dụng torchvision.transforms
def apply_augmentation(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Chuyển đổi từ BGR sang RGB và thành PIL Image
    augmented_images = []
    for _ in range(4):  # Tạo bốn ảnh augmented
        augmented_image = transform(image)
        augmented_image = augmented_image.numpy().transpose(1, 2, 0)  # Chuyển đổi từ tensor sang numpy array
        augmented_image = (augmented_image * 255).astype(np.uint8)  # Chuyển đổi từ [0, 1] sang [0, 255]
        augmented_images.append(augmented_image)
    return augmented_images

# Hàm lưu dữ liệu
def save_data(raw_folder="D:/Hocmay/test/data/"):
    dest_size = (128, 128)
    print("Bắt đầu xử lý ảnh...")
    pixels = []
    labels = []

    for folder in listdir(raw_folder):
        if folder != '.DS_Store':
            print("Folder=", folder)
            for file in listdir(raw_folder + folder):
                if file != '.DS_Store':
                    print("File=", file)
                    pixels.append(cv2.resize(cv2.imread(raw_folder + folder + "/" + file), dsize=(128, 128)))
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)

    file = open('D:/Hocmay/test/pix.data', 'wb')
    pickle.dump((pixels, labels), file)
    file.close()

# Hàm load dữ liệu
def load_data():
    file = open('D:/Hocmay/test/pix.data', 'rb')
    (pixels, labels) = pickle.load(file)
    file.close()
    return pixels, labels

# Tải dữ liệu hoặc tạo mới nếu chưa có
save_data()
X, y = load_data()
random.shuffle(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
X_train_augmented = []
y_train_augmented = []

# Áp dụng augmentation cho từng ảnh trong tập dữ liệu huấn luyện
for image, label in zip(X_train, y_train):
    augmented_images = apply_augmentation(image)
    for augmented_image in augmented_images:
        X_train_augmented.append(augmented_image)
        y_train_augmented.append(label)

# Chuyển danh sách thành mảng numpy
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# Tạo model
def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

vggmodel = get_model()

# Định nghĩa callbacks
filepath = "D:/Hocmay/test/weights-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

# Huấn luyện model
vgghist = vggmodel.fit(
    X_train_augmented, y_train_augmented,
    batch_size=64,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list
)

vggmodel.save("D:/Hocmay/test/vggmodel.h5")
