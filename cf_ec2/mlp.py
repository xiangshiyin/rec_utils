
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


class MLP:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors,
        layers,
        reg_layers
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.layers = layers # [64,32,16,8]
        self.reg_layers = reg_layers # [0,0,0,0]

    def create_model(self):
        num_layers = len(self.layers)
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the MLP embedding layer
        embedding_mlp_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1,
            name = 'embedding_mlp_User'
        )
        embedding_mlp_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1,
            name = 'embedding_mlp_Item'
        )
        
        ## the MLP branch
        latent_mlp_User = Flatten(name='flatten_mlp_User')(embedding_mlp_User(self.users_input))
        latent_mlp_Item = Flatten(name='flatten_mlp_Item')(embedding_mlp_Item(self.items_input))
        vec_mlp = Concatenate(name='concat_mlp_UserItem')([latent_mlp_User,latent_mlp_Item])
        for idx in range(1,num_layers):
            layer = Dense(
                units=self.layers[idx],
                activation='relu',
                kernel_regularizer=l2(self.reg_layers[idx]),
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
        return model