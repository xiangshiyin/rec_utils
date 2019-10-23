
import numpy as np 
# import tensorflow as tf
import keras
from keras import Model
from keras.regularizers import l2
from keras.layers import (
    Embedding, 
    Input,
    Flatten, 
    Multiply, 
    Concatenate,
    Dense
)


class NCF:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors,
        layers,
        reg_layers,
        seed
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.layers = layers # [64,32,16,8]
        self.reg_layers = reg_layers # [0,0,0,0]
        self.seed = seed

    def create_model(self):
        num_layers = len(self.layers)
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the GMF embedding layer
        embedding_gmf_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1
        )
        embedding_gmf_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1
        )
        ## create the MLP embedding layer
        embedding_mlp_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1
        )
        embedding_mlp_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1
        )

        ## the GMF branch
        latent_gmf_User = Flatten()(embedding_gmf_User(self.users_input))
        latent_gmf_Item = Flatten()(embedding_gmf_Item(self.items_input))
        vec_gmf = Multiply()([latent_gmf_User,latent_gmf_Item]) # element-wise multiply
        
        ## the MLP branch
        latent_mlp_User = Flatten()(embedding_mlp_User(self.users_input))
        latent_mlp_Item = Flatten()(embedding_mlp_Item(self.items_input))
        vec_mlp = Concatenate()([latent_mlp_User,latent_mlp_Item])
        for idx in range(1,num_layers):
            layer = Dense(
                units=self.layers[idx],
                activation='relu',
                kernel_regularizer=l2(self.reg_layers[idx])
            )
            vec_mlp = layer(vec_mlp)

        ## concatenate the output vectors from GMF and MLP branches
        vec_pred = Concatenate()([vec_gmf,vec_mlp])

        ## final prediction layer
        prediction = Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform'
        )(vec_pred)

        ## finalize the model architecture
        model = Model(
            inputs=[self.users_input,self.items_input], 
            outputs=prediction
        )
        return model










    
