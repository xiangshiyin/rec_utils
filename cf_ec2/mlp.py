
import numpy as np 
import keras
from keras import Model
from keras.regularizers import l2
from keras.optimizers import (
    Adam,
    Adamax,
    Adagrad,
    SGD,
    RMSprop
)
from keras.layers import (
    Embedding, 
    Input,
    Flatten, 
    Multiply, 
    Concatenate,
    Dense
)
from . import evaluation_grouped

class MLP:
    def __init__(
        self,
        n_users,
        n_items,
        layers_mlp,
        reg_layers_mlp=[0.,0.,0.,0.]
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.layers_mlp = layers_mlp # [64,32,16,8]
        self.n_factors_mlp = self.layers_mlp[0]//2
        self.reg_layers_mlp = reg_layers_mlp # [0,0,0,0]

    def create_model(self, path_pretrain=None):
        num_layers_mlp = len(self.layers_mlp)
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the MLP embedding layer
        embedding_mlp_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors_mlp,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = l2(self.reg_layers_mlp[0]),
            input_length = 1,
            name = 'embedding_mlp_User'
        )
        embedding_mlp_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors_mlp,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = l2(self.reg_layers_mlp[0]),
            input_length = 1,
            name = 'embedding_mlp_Item'
        )
        
        ## the MLP branch
        latent_mlp_User = Flatten(name='flatten_mlp_User')(embedding_mlp_User(self.users_input))
        latent_mlp_Item = Flatten(name='flatten_mlp_Item')(embedding_mlp_Item(self.items_input))
        vec_mlp = Concatenate(name='concat_mlp_UserItem')([latent_mlp_User,latent_mlp_Item])
        for idx in range(1,num_layers_mlp):
            layer = Dense(
                units=self.layers_mlp[idx],
                activation='relu',
                kernel_regularizer=l2(self.reg_layers_mlp[idx]),
                name='mlp_layer_{}'.format(idx)
            )
            vec_mlp = layer(vec_mlp)

        vec_pred = vec_mlp

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
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath= path_model_weights, 
            verbose=1, 
            save_best_only=True
        )
        csvlog = keras.callbacks.CSVLogger(
            filename=path_csvlog, 
            separator=',', 
            append=False
        )
        earlystop = keras.callbacks.EarlyStopping(patience=12)
        lrreduce = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.3, 
            patience=4, 
            verbose=1
        )  
        metrics2 = evaluation_grouped.metricsCallback()      
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

    