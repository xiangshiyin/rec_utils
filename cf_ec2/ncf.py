
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
from . import gmf, mlp, dataPrep


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

    def create_model(self):
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
        return model
    
    def load_pretrain_model(self, model, model_gmf, model_mlp, num_layers_mlp):
        ## get the embedding weights
        #### GMF embedding branch
        w_embedding_gmf_User = model_gmf.get_layer('embedding_gmf_User').get_weights()
        w_embedding_gmf_Item = model_gmf.get_layer('embedding_gmf_Item').get_weights()
        model.get_layer('embedding_gmf_User').set_weights(w_embedding_gmf_User)
        model.get_layer('embedding_mlp_Item').set_weights(w_embedding_gmf_Item)
        #### MLP embedding branch
        w_embedding_mlp_User = model_mlp.get_layer('embedding_mlp_User').get_weights()
        w_embedding_mlp_Item = model_mlp.get_layer('embedding_mlp_Item').get_weights()
        model.get_layer('embedding_mlp_User').set_weights(w_embedding_mlp_User)
        model.get_layer('embedding_mlp_Item').set_weights(w_embedding_mlp_Item)

        #### the MLP layers
        for idx in range(1,num_layers_mlp):
            name_mlp_layer = 'mlp_layer_{}'.format(idx)
            w_mlp_layer = model_mlp.get_layer(name_mlp_layer).get_weights()
            model.get_layer(name_mlp_layer).set_weights(w_mlp_layer)
        
        #### the output layer
        w_gmf_output = model_gmf.get_layer('output').get_weights()
        w_mlp_output = model_mlp.get_layer('output').get_weights()
        w_ncf_output = 0.5*np.concatenate(
            (w_gmf_output[0],w_mlp_output[0]),
            axis=0
        )
        b_ncf_output = 0.5*(w_gmf_output[1]+w_mlp_output[1])
        model.get_layer('output').set_weights([
            w_ncf_output,
            b_ncf_output
        ])
        return model






def get_train():
    return '','',''

#### train the NCF model
def etl():
    pass
    ## step 1: load the data
    n_users = 10000
    n_items = 50000
    n_factors_gmf = 30
    layers_mlp = [64,32,16,8]
    reg_gmf = 0.
    reg_layers_mlp = [0.,0.,0.,0.]
    learning_rate = 0.001
    flg_pretrain = ''
    filepath = ''
    filepath_gmf_pretrain = ''
    filepath_mlp_pretrain = ''
    num_epochs = 5
    batch_size = 5

    ## step 2: build the model
    ncf = NCF(
        n_users=n_users,
        n_items=n_items,
        n_factors_gmf=n_factors_gmf,
        layers_mlp=layers_mlp,
        reg_gmf=reg_gmf,
        reg_layers_mlp=reg_layers_mlp
    )
    model = ncf.create_model()
    #### compile the model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy'
    )

    ## step 3: load pretrained model
    if filepath_gmf_pretrain!='' and filepath_mlp_pretrain!='':
        #### initialize the models
        model_gmf = gmf.GMF(
            n_users=n_users,
            n_items=n_items,
            n_factors_gmf=n_factors_gmf,
            reg_gmf=reg_gmf
        ).create_model()
        model_mlp = mlp.MLP(
            n_users=n_users,
            n_items=n_items,
            layers_mlp=layers_mlp,
            reg_layers_mlp=reg_layers_mlp
        ).create_model()
        #### load the pretrained weights
        model_gmf.load_weights(filepath_gmf_pretrain)
        model_mlp.load_weights(filepath_mlp_pretrain)
        #### combine and generate the full ncf model
        model = ncf.load_pretrain_model(
            model=model,
            model_gmf=model_gmf,
            model_mlp=model_mlp,
            num_layers_mlp=len(layers_mlp)
        )



    ## step 4: train the model
    #### load the raw training data
    train = pd.read_csv(
        '',
        sep=',',
        header=0
    )
    #### generate the set of user/item/label lists
    dataset = dataPrep.Data(
        train=train,
        col_user='contact_uid',
        col_item='item',
        col_rating='freq',
        binary=True
    )
    dataset.prepTrainDNN()
    users_input = dataset.users
    items_input = dataset.items
    labels = dataset.ratings


    #### train
    hist = model.fit(
        x = [
            np.array(users_input),
            np.array(items_input)
        ],
        y = np.array(labels),
        batch_size=batch_size,
        epochs=1,
        verbose=2,
        shuffle=True
    )
    # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


    ## step 5: save the model weights
    model.save_weights(filepath=filepath) # saves weights of the model as HDF5 file
    # model.load_weights(filepath=filepath, by_name=False) # load the pretrained weights




if __name__ == "__main__":
    etl()







    
