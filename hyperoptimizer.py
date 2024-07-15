import tensorflow as tf 

from tensorflow.keras import layers, models 

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard 

from kerastuner.tuners import Hyperband 

import os 

from datetime import datetime 

 

# Sjekk og konfigurer GPU-akselerasjon 

physical_devices = tf.config.list_physical_devices('GPU') 

if len(physical_devices) > 0: 

    tf.config.experimental.set_memory_growth(physical_devices[0], True) 

 

# Definer datasettsti og annen konfigurasjon 

datasett_sti = r"C:\Users\adriva001\Downloads\Datasett_mikrometeoritter" 

 

# Dataforbehandling og dataaugmentering 

data_generator = ImageDataGenerator( 

    rescale=1.0 / 255, 

    rotation_range=20, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    horizontal_flip=True, 

    validation_split=0.2 

) 

 

# Modellbyggerfunksjon for Keras Tuner 

def model_builder(hp): 

    model = models.Sequential([ 

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)), 

        layers.MaxPooling2D((2, 2)), 

        layers.Conv2D(64, (3, 3), activation='relu'), 

        layers.MaxPooling2D((2, 2)), 

        layers.Conv2D(128, (3, 3), activation='relu'), 

        layers.MaxPooling2D((2, 2)), 

        layers.Flatten(), 

        layers.Dense(128, activation='relu'), 

        layers.Dense(2, activation='softmax') 

    ]) 

 

    # Hyperparametre som skal optimaliseres 

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) 

    hp_batch_size = hp.Choice('batch_size', values=[16, 32, 64, 128, 256]) 

 

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), 

                  loss='categorical_crossentropy', 

                  metrics=['accuracy']) 

 

    return model 

 

# Sett opp Keras Tuner 

tuner = Hyperband( 

    model_builder, 

    objective='val_accuracy', 

    max_epochs=10, 

    directory='keras_tuner_dir', 

    project_name='keras_tuner_demo' 

) 

 

# Definer tidlig stopp callback med overvåkingsmetrikk og tålmodighet 

stop_early = EarlyStopping(monitor='val_loss', patience=5) 

 

# Opprett en katalog for logging av TensorBoard 

log_katalog = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") 

tensorboard_callback = TensorBoard(log_dir=log_katalog, histogram_freq=1) 

 

# Start hyperparameteroptimaliseringen 

tuner.search( 

    data_generator.flow_from_directory( 

        datasett_sti, 

        target_size=(160, 160), 

        batch_size=32,  # Dette er et dummy-nummer, den faktiske batch-størrelsen settes av Keras Tuner 

        class_mode='categorical', 

        subset='training' 

    ), 

    validation_data=data_generator.flow_from_directory( 

        datasett_sti, 

        target_size=(160, 160), 

        batch_size=32,  # Dette er også et dummy-nummer 

        class_mode='categorical', 

        subset='validation' 

    ), 

    epochs=10, 

    callbacks=[stop_early, tensorboard_callback] 

) 

 

# Hent de beste hyperparameterne 

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 

 

print(f""" 

Optimal batch size: {best_hps.get('batch_size')} 

Optimal learning rate for the optimizer: {best_hps.get('learning_rate')} 

""") 

 

# Bygg den beste modellen basert på de beste hyperparameterne 

model = tuner.hypermodel.build(best_hps) 

 

# Trene den endelige modellen med de funnet beste hyperparameterne 

history = model.fit( 

    data_generator.flow_from_directory( 

        datasett_sti, 

        target_size=(160, 160), 

        batch_size=best_hps.get('batch_size'), 

        class_mode='categorical', 

        subset='training' 

    ), 

    validation_data=data_generator.flow_from_directory( 

        datasett_sti, 

        target_size=(160, 160), 

        batch_size=best_hps.get('batch_size'), 

        class_mode='categorical', 

        subset='validation' 

    ), 

    epochs=10,  # Du kan sette dette tallet til det du mener er passende for den endelige treningen 

    callbacks=[tensorboard_callback, stop_early] 

) 

 

# Evaluer modellen 

eval_result = model.evaluate( 

    data_generator.flow_from_directory( 

        datasett_sti, 

        target_size=(160, 160), 

        batch_size=best_hps.get('batch_size'), 

        class_mode='categorical', 

        subset='validation' 

    ), 

    verbose=2 

) 

 

print(f"[test loss, test accuracy]: {eval_result}") 

 

# Lagre modellen 

model.save('Mikrometeoritt_AI_model.h5') 

 

# Lagre klassenavnene 

class_names = list(data_generator.flow_from_directory( 

    datasett_sti, 

    subset='training' 

).class_indices.keys()) 

 

with open('class_names.txt', 'w') as file: 

    file.write("\n".join(class_names)) 
