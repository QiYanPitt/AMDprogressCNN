import numpy as np
from keras.preprocessing import image
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import inception_v3, imagenet_utils
from keras.preprocessing import image
from keras import backend as K
import parse_options
options = parse_options.parse_options()

def crop2square(img):
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))

def dummy(value):
    if value == 1:
        return [0, 0]
    elif value == 2:
        return [1, 0]
    elif value == 3:
        return [1, 1]

img = np.random.rand(224,224,3)
img_path = options.input_image 
img = crop2square(image.load_img(img_path)).resize((224, 224))
test_images = image.img_to_array(img).astype('float32') / 255
test_images = np.reshape(test_images, (1, 224, 224, 3))
with open(options.input_geno, "r") as fo:
    test_geno = fo.readlines()
test_geno = [x.strip().split() for x in test_geno]
test_geno = np.array(test_geno)

# network
def build_network():
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model2 = models.Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    flat = layers.GlobalAveragePooling2D()(base_model2.output)
    x = layers.Dense(128, activation='relu')(flat)
    output_amdState = layers.Dense(2, activation='sigmoid', name='output_amdState')(x)
    geno = layers.Input(shape=(52, ), name="geno_input") #52 SNPs
    concatenatedFeatures = layers.Concatenate(axis=1)([output_amdState, geno])
    x = layers.Dense(4, activation='relu')(concatenatedFeatures)
    output_time = layers.Dense(1, activation='sigmoid', name='output_time')(x)
    model = models.Model(inputs=[base_model2.input, geno], outputs=[output_amdState, output_time])
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    base_model2.trainable = False
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    return model

model = build_network()
model.load_weights(options.weights + "/analysis_cat" + options.cutoff_yr + "/model_finetune_imgNgeno2amdState2time.h5")
y_test_pred = model.predict([test_images, test_geno])

f = open(options.out_file, "w")
adv = y_test_pred[0][0][0] * y_test_pred[0][0][1]
inter = y_test_pred[0][0][0] * (1-y_test_pred[0][0][1])
no = (1-y_test_pred[0][0][0]) * (1-y_test_pred[0][0][1])
f.write("%s\t%s\t%s\t%s\n%f\t%f\t%f\t%f\n" % ('prob_abv_cutoff', 'prob_adv_AMD', 'prob_inter_AMD', 'prob_no_AMD', y_test_pred[1], adv, inter, no))
f.close()
