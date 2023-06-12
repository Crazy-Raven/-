from tensorflow.keras.layers import *


# 基于 vgg16 的 segnet 编码器
def segnet_encoder_vgg16(height=416, width=416):
    img_input = Input(shape=(height, width, 3))

    # block1
    # 416,416,3 -- 208,208,64
    x = Conv2D(64, 3, padding='same', activation='relu', name='b1_c1')(img_input)
    x = Conv2D(64, 3, padding='same', activation='relu', name='b1_c2')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b1_pool')(x)
    out1 = x

    # block2
    # 208,208,64 -- 104,104,128
    x = Conv2D(128, 3, padding='same', activation='relu', name='b2_c1')(x)
    x = Conv2D(128, 3, padding='same', activation='relu', name='b2_c2')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b2_pool')(x)
    out2 = x

    # block3
    # 104,104,128 -- 52,52,256
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c1')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c2')(x)
    x = Conv2D(256, 3, padding='same', activation='relu', name='b3_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b3_pool')(x)
    out3 = x

    # block4
    # 52,52,256 -- 26,26,512
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c1')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c2')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b4_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b4_pool')(x)
    out4 = x

    # block5
    # 26,26,512 -- 13,13,512
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c1')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c2')(x)
    x = Conv2D(512, 3, padding='same', activation='relu', name='b5_c3')(x)
    x = MaxPooling2D((2, 2), strides=2, name='b5_pool')(x)
    out5 = x

    return img_input, out3


from tensorflow.keras.layers import *


# segnet的解码器
def segnet_decoder(feature, n_classes):
    # 直接进行上采样时会出现一些问题，这里先Zeropadding
    # 26,26,512 -- 26,26,512
    x = ZeroPadding2D(1)(feature)  # 26,26,512 -- 28,28,512
    x = Conv2D(512, 3, padding='valid')(x)  # 28,28,512 -- 26,26,512
    x = BatchNormalization()(x)

    # 上采样 3 次(编码器总共编码5次，每次图像缩小一半，只用第3次的结果)
    # 1/16 -- 1/8 ; 26,26,512 -- 52,52,256
    # 1/8 -- 1/4  ; 52,52,256 -- 104,104,128
    # 1/4 -- 1/2  ; 104,104,128 -- 208,208,64
    filters = [256, 128, 64]
    for i, filter in enumerate(filters):
        x = UpSampling2D(2, name='Up_' + str(i + 1))(x)
        x = ZeroPadding2D(1)(x)
        x = Conv2D(filter, 3, padding='valid')(x)
        x = BatchNormalization()(x)

    # 208,208,64 -- 208,208,n_classes
    out = Conv2D(n_classes, 3, padding='same')(x)
    return out


# from encoders import *
from tensorflow.keras.models import Model


# 创建 segnet 模型
def build_segnet(n_classes, encoder_type='vgg16', input_height=416, input_width=416):
    # 1.获取encoder的输出 (416,416,3--26,26,512)
    if encoder_type == 'vgg16':
        img_input, feature = segnet_encoder_vgg16(input_height, input_width)
    elif encoder_type == 'MobilenetV1_1':
        img_input, feature = segnet_encoder_MobilenetV1_1(input_height, input_width)
    elif encoder_type == 'MobilenetV1_2':
        img_input, feature = segnet_encoder_MobilenetV1_2(input_height, input_width)
    else:
        raise RuntimeError('segnet encoder name is error')

    # 2.获取decoder的输出 (26,26,512--208,208,n_classes)
    out = segnet_decoder(feature, n_classes)
    # 3.结果Reshape (208*208,n_classes)
    # out = Reshape((int(input_height / 2) * int(input_height / 2), -1))(out)
    out = Softmax()(out)
    # 4.创建模型
    model = Model(img_input, out)

    return model


if __name__ == '__main__':
    model = build_segnet(10, 'vgg16', 64, 64)
    model.summary()
