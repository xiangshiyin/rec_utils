
import numpy as np 
import pandas as pd
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
from . import (
    gmf, 
    mlp, 
    dataPrep,
    evaluation_grouped
)


class NCF:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors_gmf,
        layers_mlp,
        reg_gmf=0.,
        reg_layers_mlp=[0.,0.,0.,0.]
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors_gmf = n_factors_gmf
        self.layers_mlp = layers_mlp # [64,32,16,8]
        self.n_factors_mlp = self.layers_mlp[0]//2
        self.reg_gmf = reg_gmf
        self.reg_layers_mlp = reg_layers_mlp # [0,0,0,0]

    def create_model(self, path_pretrain=None):
        num_layers_mlp = len(self.layers_mlp)
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the GMF embedding layer
        embedding_gmf_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors_gmf,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = l2(self.reg_gmf),
            input_length = 1,
            name = 'embedding_gmf_User'
        )
        embedding_gmf_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors_gmf,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = l2(self.reg_gmf),
            input_length = 1,
            name = 'embedding_gmf_Item'
        )
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

        ## the GMF branch
        latent_gmf_User = Flatten(name='flatten_gmf_User')(embedding_gmf_User(self.users_input))
        latent_gmf_Item = Flatten(name='flatten_gmf_Item')(embedding_gmf_Item(self.items_input))
        vec_gmf = Multiply(name='multiply_gmf_UserItem')([latent_gmf_User,latent_gmf_Item]) # element-wise multiply
        
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

        ## concatenate the output vectors from GMF and MLP branches
        vec_pred = Concatenate(name='concat_gmf_mlp')([vec_gmf,vec_mlp])

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
        
        ## load pretrain ncf model
        if path_pretrain:
            model.load_weights(path_pretrain)
        self.model = model
    
    def load_pretrain_model(self, model_gmf, model_mlp, num_layers_mlp):
        ## get the embedding weights
        #### GMF embedding branch
        w_embedding_gmf_User = model_gmf.get_layer('embedding_gmf_User').get_weights()
        w_embedding_gmf_Item = model_gmf.get_layer('embedding_gmf_Item').get_weights()
        self.model.get_layer('embedding_gmf_User').set_weights(w_embedding_gmf_User)
        self.model.get_layer('embedding_mlp_Item').set_weights(w_embedding_gmf_Item)
        #### MLP embedding branch
        w_embedding_mlp_User = model_mlp.get_layer('embedding_mlp_User').get_weights()
        w_embedding_mlp_Item = model_mlp.get_layer('embedding_mlp_Item').get_weights()
        self.model.get_layer('embedding_mlp_User').set_weights(w_embedding_mlp_User)
        self.model.get_layer('embedding_mlp_Item').set_weights(w_embedding_mlp_Item)

        #### the MLP layers
        for idx in range(1,num_layers_mlp):
            name_mlp_layer = 'mlp_layer_{}'.format(idx)
            w_mlp_layer = model_mlp.get_layer(name_mlp_layer).get_weights()
            self.model.get_layer(name_mlp_layer).set_weights(w_mlp_layer)
        
        #### the output layer
        w_gmf_output = model_gmf.get_layer('output').get_weights()
        w_mlp_output = model_mlp.get_layer('output').get_weights()
        w_ncf_output = 0.5*np.concatenate(
            (w_gmf_output[0],w_mlp_output[0]),
            axis=0
        )
        b_ncf_output = 0.5*(w_gmf_output[1]+w_mlp_output[1])
        self.model.get_layer('output').set_weights([
            w_ncf_output,
            b_ncf_output
        ])
    
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


