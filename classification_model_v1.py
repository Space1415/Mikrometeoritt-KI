import tensorflow as tf 

from tensorflow.keras import layers 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

 

# Sjekk og konfigurer GPU-akselerasjon 

physical_devices = tf.config.list_physical_devices('GPU') 

if len(physical_devices) > 0: 

    tf.config.experimental.set_memory_growth(physical_devices[0], True) 

 

# Definer datasettsti og annen konfigurasjon 

datasett_sti = r"C:\Users\adriva001\Downloads\Datasett_mikrometeoritter" 

batch_storrelse = 32 

epoker = 25 

 

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

 

# Modellarkitektur 

model = tf.keras.Sequential([ 

    layers.Conv2D(256, (3, 3), activation='relu', input_shape=(160, 160, 3), padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    layers.Conv2D(384, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(384, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'), 

    layers.BatchNormalization(), 

    layers.MaxPooling2D((2, 2)), 

 

    layers.Flatten(), 

    layers.Dense(512, activation='relu'), 

    layers.BatchNormalization(), 

    layers.Dropout(0.5), 

    layers.Dense(256, activation='relu'), 

    layers.BatchNormalization(), 

    layers.Dropout(0.3), 

    layers.Dense(2, activation='softmax')  # 2 klasser) 

]) 

 

# Kompilér modellen med Adam-optimalisator 

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy']) 

 

# Trening av modellen 

model.fit(trenings_generator, validation_data=validerings_generator, epochs=epoker) 

 

# Evaluer modellen 

test_loss, test_acc = model.evaluate(validerings_generator, verbose=2) 

print('Testnøyaktighet:', test_acc) 

 

# Lagre modellen 

model.save('Mikrometeoritt_AI_model.keras') 

 

# Lagre klassenavnene 

class_names = list(trenings_generator.class_indices.keys()) 

with open('class_names.txt', 'w') as file: 

    file.write("\n".join(class_names)) 
