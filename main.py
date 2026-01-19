import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'D:/test/CAPSTONE GITAM/pneumonia_dataset_kaggle/chest_xray/chest_xray'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(f'{DATA_DIR}/train', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
validation_generator = validation_test_datagen.flow_from_directory(f'{DATA_DIR}/val', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
test_generator = validation_test_datagen.flow_from_directory(f'{DATA_DIR}/test', target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

print("Data generators created successfully!")


print("\nBuilding model with pre-trained weights...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


print("\n--- STAGE 1: Training the custom head ---")
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history_stage1 = model.fit(
    train_generator,
    epochs=5,  
    validation_data=validation_generator
)


print("\n--- STAGE 2: Fine-tuning the top layers ---")
base_model.trainable = True

fine_tune_at = 143 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_pneumonia_finetuned.keras', save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_stage2 = model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

print("\nTraining finished!")
