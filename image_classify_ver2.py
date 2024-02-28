### 필요한 라이브러리/함수 import
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# target image size 설정
HEIGHT = 224
WIDTH = 224

class DataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    __train_dir__   = '/content/drive/MyDrive/KDT/05_딥러닝/02_컴퓨터비전/02_Data파일/mouse/train'
    __val_dir__     = '/content/drive/MyDrive/KDT/05_딥러닝/02_컴퓨터비전/02_Data파일/mouse/val'

    def __init__(self,         # 클래스 생성자
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=0,
                width_shift_range=0.0,
                height_shift_range=0.0,
                brightness_range=None,
                shear_range=0.0,
                zoom_range=0.0,
                channel_shift_range=0.0,
                fill_mode="nearest",
                cval=0.0,
                horizontal_flip=False,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0,
                interpolation_order=1,
                dtype=None,):
        super().__init__(self,  # 부모 클래스 생성자 호출
                         featurewise_center=False,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         zca_whitening=False,
                         zca_epsilon=1e-6,
                         rotation_range=0,
                         width_shift_range=0.0,
                         height_shift_range=0.0,
                         brightness_range=None,
                         shear_range=0.0,
                         zoom_range=0.0,
                         channel_shift_range=0.0,
                         fill_mode="nearest",
                         cval=0.0,
                         horizontal_flip=False,
                         vertical_flip=False,
                         rescale=None,
                         preprocessing_function=None,
                         data_format=None,
                         validation_split=0.0,
                         interpolation_order=1,
                         dtype=None,)
        
    # 데이터 증식 및 조건 설정
    def make_datagen(self, augmentation):
        self.augmentation = augmentation
        if self.augmentation:
            return tf.keras.preprocessing.image.ImageDataGenerator(
                        # scaling
                        rescale=1./255,

                        # 하이퍼 파라미터 설정
                        rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest')
        else:
            return tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    # flow_from_directory method overide
    def make_datagen_flow_from_dir(self, datagen, gubun):
        self.datagen = datagen
        self.gubun = gubun

        data_generator = self.datagen.flow_from_directory(
            directory = self.__train_dir__ if self.gubun == 'train' else self.__val_dir__,
            target_size = (HEIGHT, WIDTH),
            class_mode = 'categorical',
            batch_size = 20,
            seed=0)
        return data_generator
    
class Model:

    # 생성 모델 멤버변수
    pretrained_model = finetuned_model = ''

    ### 학습 결과를 저장할 디렉토리 생성
    # __model_dir__ = './data/model'
    __model_dir__   = '/content/drive/MyDrive/KDT/05_딥러닝/02_컴퓨터비전/02_Data파일/mouse/model'
    __test_dir__    = '/content/drive/MyDrive/KDT/05_딥러닝/02_컴퓨터비전/02_Data파일/mouse/test'

    # 출력 units 설정
    __units__ = 2

    def __init__(self, model): # 클래스 생성자
        self.model = model

        ### 랜덤 시드 설정
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)
        self.glorot_uniform=tf.keras.initializers.GlorotUniform(seed=0)

    # pretrained_model create
    def make_pretrained_model(self):
        if self.model == 'vgg16':
            self.pretrained_model = tf.keras.applications.vgg16.VGG16(
                include_top = False,
                weights = 'imagenet',
                input_shape = (HEIGHT, WIDTH, 3))
        elif self.model == 'resnet':
            self.pretrained_model = tf.keras.applications.resnet_v2.ResNet152V2(
                include_top = False,
                weights = 'imagenet',
                input_shape = (HEIGHT, WIDTH, 3))

        # freezing
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        # model_summary
        self.pretrained_model.summary()

    # fine-tuning_model create
    def make_finetuned_model(self):
        # 모델구성
        self.finetuned_model = tf.keras.Sequential()
        self.finetuned_model.add(self.pretrained_model)
        self.finetuned_model.add(tf.keras.layers.Flatten())
        self.finetuned_model.add(tf.keras.layers.Dense(units=256,
                                                       activation='relu',
                                                       kernel_initializer=self.glorot_uniform))
        self.finetuned_model.add(tf.keras.layers.Dropout(rate=0.5))
        self.finetuned_model.add(tf.keras.layers.Dense(units=2,
                                                       activation='softmax',
                                                       kernel_initializer=self.glorot_uniform))
        # model_summary
        self.finetuned_model.summary()

    def training(self, save_filename, train_generated, val_generated):
        self.save_filename = save_filename
        self.train_generated = train_generated
        self.val_generated = val_generated

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                           verbose = 1,
                                                           patience = 5)
        # 최적의 학습 결과를 저장하기 위한 ModelCheckpoint 설정
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.__model_dir__ + '/' + self.save_filename,
            monitor = 'val_loss',
            verbose = 1,
            save_best_only = True)

        # 모델 컴파일
        self.finetuned_model.compile(loss='categorical_crossentropy',
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                                     metrics=['accuracy'])

        # 학습
        history = self.finetuned_model.fit(self.train_generated,
                                           validation_data=self.val_generated,
                                           epochs=1000,
                                           callbacks = [self.early_stop, self.checkpoint])
        return history

    ## 학습 결과 시각화
    def plot(self, history):
        self.history = history

        fig, ax = plt.subplots(1, 2, figsize=(18,8))

        # accuracy, loss 추출 -> 그래프 y축 설정
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        # 그래프 x축
        epochs = np.arange(1, len(train_acc)+1)

        # accuracy 시각화
        ax[0].plot(epochs, train_acc, c='b', label='Training acc')
        ax[0].plot(epochs, val_acc, c='g', label='Validation acc')
        ax[0].set_title('[Accuracy]', size=20)
        ax[0].set_xlabel('epochs') # X축
        ax[0].set_ylabel('accuracy', size=16) # y축
        ax[0].legend() # 범례 생성

        # loss 시각화
        ax[1].plot(epochs, train_loss, c='r', label='Training loss')
        ax[1].plot(epochs, val_loss, c='m', label='Validation loss')
        ax[1].set_title('[Loss]', size=20)
        ax[1].set_xlabel('epochs') # X축
        ax[1].set_ylabel('loss', size=16) # y축
        ax[1].legend() # 범례 생성

        plt.show()

    # 테스트 이미지 분류
    def classify_img(self):
        image_path = []
        for name in sorted(os.listdir(self.__test_dir__)):
            if name.find('.') > 0:
                image_path.append(self.__test_dir__ + '/' + name)
        print(f'평가용 이미지 경로 확인 :\n{image_path}')

        # 평가용 이미지를 이용한 평가 진행
        for image in image_path:
            img = Image.open(image)
            img_array = np.array(img.resize((HEIGHT,WIDTH)))

            # 이미지 픽셀에 대한 rescaling
            X = img_array/255
            # 이미지는 3차원 --> 4차원으로 변환 : batch_size=1 --> (1,224,224,3)
            X = X.reshape(1,HEIGHT,WIDTH,3)

            # 정답 label 설정
            label = ['mouse_part','product']

            # 평가용 데이터를 이용한 예측 : 2차원 배열로 나온다.
            pred = self.finetuned_model.predict(X)
            # 예측값(확률)중 가장 높은 인덱스 추출
            idx = np.argmax(pred[0])    # pred가 2차원 배열이기 때문에 인덱싱을 통해 꺼내준다.
            # 정답 --> 숫자 대신 폴더명('mouse_part','product')으로 출력
            y = label[idx]
            print(f'모델의 예측 label : {y}')
            # 예측확률 추출
            print(f'정확도 : {np.max(pred)}')

            # 이미지 출력
            plt.imshow(img_array)
            plt.axis('off')
            plt.show()

# DataGenerator 객체 생성
datagen = DataGenerator()
