import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from datetime import datetime 

from tensorflow.keras.callbacks import EarlyStopping 

 

# Sjekk og konfigurer GPU-akselerasjon 

physical_devices = tf.config.list_physical_devices('GPU') 

if len(physical_devices) > 0: 

    # Spesifiser navnet på den ønskede GPU-en (f.eks. '/device:GPU:0' for den første GPU-en) 

    gpu_name = '/device:GPU:0' 

    tf.config.set_visible_devices(physical_devices, gpu_name) 

    for device in physical_devices: 

        tf.config.experimental.set_memory_growth(device, True) 

                      

# Definer datasettsti og annen konfigurasjon 

datasett_sti = r"C:\Users\adriva001\Downloads\Datasett_mikrometeoritter" 

batch_storrelse = 240 

epoker = 320 

 

# Dataforbehandling og dataaugmentering 

data_generator = ImageDataGenerator( 

    rescale=1.0 / 255, 

    rotation_range=20, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    horizontal_flip=True, 

    validation_split=0.2 

) 

 

trenings_generator = data_generator.flow_from_directory( 

    datasett_sti, 

    target_size=(160, 160), 

    batch_size=batch_storrelse, 

    class_mode='categorical', 

    subset='training' 

) 

 

validerings_generator = data_generator.flow_from_directory( 

    datasett_sti, 

    target_size=(160, 160), 

    batch_size=batch_storrelse, 

    class_mode='categorical', 

    subset='validation' 

) 

 

model = models.Sequential([ 

    # Første blokk 

    layers.Conv2D(512, (3, 3), activation='relu', input_shape=(160, 160, 3), padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    # Andre blokk 

    layers.Conv2D(768, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(768, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    # Tredje blokk 

    layers.Conv2D(1024, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(1024, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    # Fjerde blokk 

    layers.Conv2D(1024, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(1024, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    # Fullstendig tilkoblet del 

    layers.Flatten(), 

    layers.Dense(1024, activation='relu'), 

    layers.BatchNormalization(), 

    layers.Dropout(0.5), 

    layers.Dense(512, activation='relu'), 

    layers.BatchNormalization(), 

    layers.Dropout(0.4), 

 

    # Utdata 

    layers.Dense(2, activation='softmax')  # Anta at det er 2 klasser 

]) 

 

# Kompilér modellen med Adam-optimalisator og ekstra metrikker (inkludert F1-score) 

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), tf.keras.metrics.BinaryF1Score()]) 

 

# Opprett en katalog for logging av TensorBoard 

log_katalog = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") 

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_katalog, histogram_freq=1) 

 

# Definer tidlig stopp callback med overvåkingsmetrikk og tålmodighet 

tidlig_stopp = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) 

 

# Trening av modellen med TensorBoard og tidlig stopp callback 

model.fit(trenings_generator, validation_data=validerings_generator, epochs=epoker, callbacks=[tensorboard_callback, tidlig_stopp]) 

 

# Evaluer modellen med ekstra metrikker 

test_loss, test_acc, test_precision, test_recall, test_auc, test_f1 = model.evaluate(validerings_generator, verbose=2) 

print('Testnøyaktighet:', test_acc) 

print('Precision:', test_precision) 

print('Recall:', test_recall) 

print('AUC:', test_auc) 

print('F1 Score:', test_f1) 

 

# Lagre modellen 

model.save('Mikrometeoritt_AI_model.keras') 

 

# Lagre klassenavnene 

class_names = list(trenings_generator.class_indices.keys()) 

with open('class_names.txt', 'w') as file: 

    file.write("\n".join(class_names)) 

 
