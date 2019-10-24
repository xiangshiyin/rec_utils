
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


class GMF:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

    def create_model(self):
        ## create the input layer
        self.users_input = Input(shape=(1,), dtype='int32', name='user_input')
        self.items_input = Input(shape=(1,), dtype='int32', name='item_input')
        ## create the GMF embedding layer
        embedding_gmf_User = Embedding(
            input_dim = self.n_users,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
            input_length = 1,
            name = 'embedding_gmf_User'
        )
        embedding_gmf_Item = Embedding(
            input_dim = self.n_items,
            output_dim = self.n_factors,
            embeddings_initializer = 'truncated_normal',
            embeddings_regularizer = keras.regularizers.l2(0.),
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
        return model