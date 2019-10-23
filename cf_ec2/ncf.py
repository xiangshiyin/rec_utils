
import numpy as np 
# import tensorflow as tf
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


class NCF:
    def __init__(
        self,
        n_users,
        n_items,
        n_factors,
        layers,
        reg_layers,
        seed=123
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

def GFM():
    def __init__(self):
        pass

def MLP():
    def __init__(self):
        pass

def get_train():
    return '','',''

#### train the NCF model
def etl():
    pass
    ## step 1: load the data
    n_users = 10000
    n_items = 50000
    n_factors = 30
    layers = [64,32,16,8]
    reg_layers = [0,0,0,0]
    learning_rate = 0.001
    flg_pretrain = ''
    filepath = ''
    num_epochs = 5
    batch_size = 5

    ## step 2: build the model
    ncf = NCF(
        n_users=n_users,
        n_items=n_items,
        n_factors=n_factors,
        layers=layers,
        reg_layers=reg_layers
    )
    model = ncf.create_model()
    #### compile the model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy'
    )

    ## step 3: load pretrained model
    if flg_pretrain != '':
        pass
        # model = load_pretrained_model()

    ## step 4: train the model
    users_input, items_input, labels_input = get_train()
    #### train
    model.fit(
        x = [np.array(users_input),np.array(items_input)],
        y = np.array(labels_input),
        batch_size=batch_size,
        epochs=1,
        verbose=2,
        shuffle=True
    )


    ## step 5: save the model weights
    model.save_weights(filepath=filepath) # saves weights of the model as HDF5 file
    # model.load_weights(filepath=filepath, by_name=False) # load the pretrained weights




if __name__ == "__main__":
    etl()







    
