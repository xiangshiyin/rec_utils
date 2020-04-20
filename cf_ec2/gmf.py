
import numpy as np 
import tensorflow as tf
# import keras
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import (
    Adam,
    Adamax,
    Adagrad,
    SGD,
    RMSprop
)
from tensorflow.keras.layers import (
    Embedding, 
    Input,
    Flatten, 
    Multiply, 
    Concatenate,
    Dense
)
from . import evaluation_grouped

class GMF:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors_gmf,
        reg_gmf=0.
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors_gmf = n_factors_gmf
        self.reg_gmf = reg_gmf

    def create_model(self, path_pretrain=None):
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the GMF embedding layer
        embedding_gmf_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors_gmf,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(self.reg_gmf),
            input_length = 1,
            name = 'embedding_gmf_User'
        )
        embedding_gmf_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors_gmf,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(self.reg_gmf),
            input_length = 1,
            name = 'embedding_gmf_Item'
        )

        ## the GMF branch
        latent_gmf_User = Flatten(name='flatten_gmf_User')(embedding_gmf_User(self.users_input))
        latent_gmf_Item = Flatten(name='flatten_gmf_Item')(embedding_gmf_Item(self.items_input))
        vec_gmf = Multiply(name='multiply_gmf_UserItem')([latent_gmf_User,latent_gmf_Item]) # element-wise multiply
        vec_pred = vec_gmf

        ## final prediction layer
        prediction = Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='output'
        )(vec_pred)

        ## finalize the model architecture
        model = Model(
            inputs=[self.users_input,self.items_input], 
            outputs=prediction
        )
        
        ## load pretrain model
        if path_pretrain:
            model.load_weights(path_pretrain)
        self.model = model
    
    def compile(self,learning_rate):
        ## compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, dataset, batch_size, num_epochs, path_model_weights, path_csvlog):
        ## create the callback metrics
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath= path_model_weights,  
            monitor='val_loss',
            verbose=1, 
            save_best_only=True
        )
        csvlog = tf.keras.callbacks.CSVLogger(
            filename=path_csvlog, 
            separator=',', 
            append=False
        )
        earlystop = tf.keras.callbacks.EarlyStopping(patience=12)
        lrreduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.3, 
            patience=4, 
            verbose=1
        )        
        metrics2 = evaluation_grouped.metricsCallback(batch_size=batch_size,log_steps=100)
        ## fit the model
        hist = self.model.fit(
            x = [
                np.array(dataset.users),
                np.array(dataset.items)
            ],
            y = np.array(dataset.ratings),
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=2,
            shuffle=True,
            callbacks=[metrics2,checkpoint,csvlog,earlystop,lrreduce],
            validation_data=(
                [
                    np.array(dataset.users_test),
                    np.array(dataset.items_test)
                ],
                np.array(dataset.ratings_test)
            )
        )  
        return hist      

    