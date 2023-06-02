# basic
import pickle as pkl
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

# AI 
from classification_models.keras import Classifiers
import tensorflow.keras as keras
import tensorflow as tf

# images
from PIL import Image

class NoisyEnsemble:
    def __init__(self, train_df = None,
                 test_df = None,
                 noise_level=0.15, 
                 number_cnn=15, 
                 epochs=10, 
                 h5_path="NoisyEnsemble_weights/",
                 run_properties_path="NoisyEnsemble_runs/",
                 test_results_out="NoisyEnsemble_test_results.csv",
                 test_tiles_out="NoisyEnsemble_test_tile_predictions.csv",
                 noisy_validation=False,
                 save_best = True,
                 batch_size = 24,
                 image_dim = (512,512)):
        """
        Class for training and testing noisy ensembles on binary class image data.
        
        !DataFrame inputs need to contain columns for "patient id", "path", "label"!
        patient id : ID for each tile/patch from the same patient.
        path       : Path to tile/patch file, should be able to be opened by Pillow.Image
        label      : Binary label, should be 1 or 0.
        
        For a full run with training and test initialize with both train_df and test_df.
        Run NoisyEnsembl.train_test()
        
        For training initialize with training dataframe.
        Run NoisyEnsemble.train_full()
        
        For testing initialize with test dataframe.
        Run NoisyEnsemble.test_ensemble()
        
        
        Number of CNNs should be higher in training than the number of unique patient 
        to ensure enough validation splits can be created.
        
        EXAMPLE: 
        'Import NoisyEnsemble
        
        NE = NoisyEnsemble.NoisyEnsemble(train_df=your_train_df,test_df=your_test_df)
        NE.train_test()'
        
        train_df         : pandas.DataFrame
        test_df          : pandas.DataFrame
        noise_level      : bool
        number_cnn       : int
        epochs           : int
        h5_path          : str
        run_properties   : str
        test_results_out : str
        noisy_validation : bool
        save_best        : True
        batch_size       : int
        image_dim        : (int,int)
        """
        self.train_df = train_df
        self.test_df = test_df.copy(deep=True)
        self.noise_level = noise_level
        self.number_cnn = number_cnn
        
        self.h5_path = h5_path
        self.run_props = run_properties_path
        self.test_results_out = test_results_out
        self.test_tiles_out = test_tiles_out
        
        self.noisy_validation = noisy_validation
        self.save_best = save_best
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.epochs = epochs
        
        self.splits = []
        self.patient_list = []
        self.generators_train = []
        self.generators_val = []
        
        # create Folders
        Path(self.h5_path).mkdir(parents=True, exist_ok=True)
        Path(self.run_props).mkdir(parents=True, exist_ok=True)
        if "/" in self.test_results_out:
            out_path = "/".join(self.test_results_out.split("/")[:-1])
            Path(self.test_results_out).mkdir(parents=True, exist_ok=True)
        if "/" in self.test_tiles_out:
            out_path = "/".join(self.test_tiles_out.split("/")[:-1])
            Path(self.test_results_out).mkdir(parents=True, exist_ok=True)
    
    def train_test(self):
        """
        Full training and testing.
        Needs train_df AND test_df.
        """
        print("Running full run: Training + Test")
        self.train_val_splits()
        self.prepare_noisy_generators()
        self.train_ensemble()
        
        self.test_ensemble()
        
    def full_training(self):
        """
        Full training, needs train_df.
        """
        print("Start Training Process")
        self.train_val_splits()
        self.prepare_noisy_generators()
        self.train_ensemble()
        
    
    def get_model(self, model_type="ResNet18"):
        """
        Load Model Architecture (ResNet18, add own architecture if desired).
        Default ResNet18.
        Load ImageNet pretrained weights.
        """
        # load architecture and weights
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape=(512,512,3), weights='imagenet', include_top=False)
        
        # add own top
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(2, activation='softmax')(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        
        # compile
        model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy',"AUC"])
        
        return model
    
    def train_val_splits(self):
        """
        Takes input df (self.train_df) to create the defined 
        number (self.number_cnn) of validation sets of equal size.
        """
        
        # number of patients needs to be larger than number of cnn
        
        # unique patients:
        self.patient_list = self.train_df["patient id"].unique().tolist()
        np.random.shuffle(self.patient_list)
        
        # size of validation splits
        patlist_len = len(self.patient_list)
        split_size = patlist_len//self.number_cnn
        
        # extract splits and save to self
        for i in range(split_size,split_size*self.number_cnn+1,split_size):
            self.splits.append(self.patient_list[i-split_size:i]) 
    
    def load_images(self,image_list):
        """
        image_list : str, list of filepaths to images that should be loaded.
        
        Returns Image data as np.array loaded from input list. Converts to RGB.
        """
        imges = []
        for img in image_list:
            imges.append(np.asarray(Image.open(img).convert("RGB")))

        return np.array(imges)
    
    def prepare_noisy_generators(self):
        """
        Uses self.splits, self.train_df, self.number_cnn.
        Sets self.generators_train, self.generators_val.
        Based on self.splits will for either training set or training and validations
        create data generators for tensorflow, where each patient has only one label
        which the has self.noise_level flipped labels.
        """
        for i in range(self.number_cnn):
            # load validation patient list
            validation_pats = self.splits[i]
            split_train_df = self.train_df[~self.train_df["patient id"].isin(validation_pats)]
            split_val_df   = self.train_df[self.train_df["patient id"].isin(validation_pats)]
            
            # take only one label per patient for training patients
            temp = []
            temp_pat_list = split_train_df["patient id"].unique().tolist()
            # first half of patients: take only label 1, second half: take only label 0
            for id, pat in enumerate(temp_pat_list):
                if id <= len(temp_pat_list)//2:
                    temp.append(self.train_df[(self.train_df["patient id"] == pat) & 
                                              (self.train_df["label"] == 1)])
                else:
                    temp.append(self.train_df[(self.train_df["patient id"] == pat) & 
                                              (self.train_df["label"] == 0)])
            split_train_df = pd.concat(temp)
            
            #add noise
            flip = []
            temp_pat_list = split_train_df["patient id"].unique().tolist()
            for pat in temp_pat_list:
                to_change = self.train_df[self.train_df["patient id"]==pat]
                to_change = to_change.sample(frac=self.noise_level).index
                flip.append(list(to_change))
                
            flip = [item for sublist in flip for item in sublist]
            noisy = []
            for id,row in split_train_df.iterrows():
                if id in flip:
                    noisy.append(abs(1-row["label"]))
                else:
                    noisy.append(row["label"])
            split_train_df["label"] = noisy
            
            
            # if noisy validation is desired repeat for validation
            if self.noisy_validation:
                # one class per patient
                temp = []
                temp_pat_list = split_val_df["patient id"].unique().tolist()
                for id, pat in enumerate(temp_pat_list):
                    if id <= len(temp_pat_list)//2:
                        temp.append(self.train_df[(self.train_df["patient id"] == pat) & 
                                                  (self.train_df["label"] == 1)])
                    else:
                        temp.append(self.train_df[(self.train_df["patient id"] == pat) & 
                                                  (self.train_df["label"] == 0)])
                split_val_df = pd.concat(temp)
            
                # add label noise
                flip = []
                temp_pat_list = split_val_df["patient id"].unique().tolist()
                for pat in temp_pat_list:
                    to_change = self.train_df[self.train_df["patient id"]==pat]
                    to_change = to_change.sample(frac=self.noise_level).index
                    flip.append(list(to_change))
                    
                flip = [item for sublist in flip for item in sublist]
                noisy = []
                for id,row in split_val_df.iterrows():
                    if id in flip:
                        noisy.append(abs(1-row["label"]))
                    else:
                        noisy.append(row["label"])
                split_val_df["label"] = noisy
            
            # define generators
            
            generator_train = Flow_from_list(split_train_df["path"].tolist(),
                                             split_train_df["label"].tolist(),
                                             self.batch_size,
                                             self.image_dim,
                                             3,2,True)
            
            
            generator_val   = Flow_from_list(split_val_df["path"].tolist(),
                                             split_val_df["label"].tolist(),
                                             self.batch_size,
                                             self.image_dim,
                                             3,2,True)
            self.generators_train.append(generator_train)
            self.generators_val.append(generator_val)
            
    
    def train_ensemble(self):
        """
        Based on self.generators_train/val trains networks for self.epoch 
        epochs and a batchsize of self.batch_size.
        Saves only the best model (only weights) to self.h5_paths.
        Saves patients that were part of the validation set to file.
        """
        print("Training Ensemble...")
        for i in range(self.number_cnn):
            model = None
            # get data generator for training and validation for split
            gen_train = self.generators_train[i]
            gen_val   = self.generators_val[i]
            
            #savepath
            path_out = f"{self.h5_path}ensemble_model_{i}.h5"
            
            # callback, save only best
            mcc = [tf.keras.callbacks.ModelCheckpoint(filepath=path_out,
                                                     save_weights_only=True,
                                                     monitor='val_accuracy',
                                                     mode='max',
                                                     save_best_only=True)]
            if not self.save_best:
                mcc = None
            
            # get model
            model = self.get_model()
            
            # train
            history = model.fit(gen_train, 
                                validation_data=gen_val,
                                batch_size=self.batch_size, 
                                verbose=0, 
                                epochs=self.epochs,
                                callbacks=mcc)
            
            # save patients in validation
            with open(f"{self.run_props}ensemble_model_{i}.txt","w") as f:
                f.write("Validation Set:\n")
                for pat in self.splits[i]:
                    f.write(pat + "\n")
            
    def test_ensemble(self):
        """
        Loads h5 models into default architecture (ResNet18) as defined by self.h5_paths.
        Loads test data from self.test_df.
        Predicts test data and calculates the ensemble prediction and agreement for each input image.
        Predicts also the ensemble accuracy and retained tiles for each agreement cutoff.
        Saves results to csvs to self.test_results_out and self.test_tiles_out.
        """
        print("Testing...")
        # test_files_paths
        len_test = self.test_df.shape[0]
        image_paths = self.test_df["path"].tolist()
        
        # get model_paths
        models_paths = glob(self.h5_path + "*.h5")
        
        ensemble_predictions = []
        for path in models_paths:
            # load model
            model = self.get_model()
            model.load_weights(path)
            
            # predict self.batch_size images at the same time
            predictions = []
            for i in range(self.batch_size,len_test,self.batch_size):
                imges = image_paths[i-self.batch_size:i]
                imges = self.load_images(imges)
                pred = model(imges)
                # get only predicted label
                predictions += [np.argmax(x) for x in pred]
                last_step = i
            # predict final incomplete batch if exists
            if len(predictions) != len_test:
                imges = image_paths[last_step:]
                imges = self.load_images(imges)
                pred = model(imges)
                # get only predicted label
                predictions += [np.argmax(x) for x in pred]
                
            ensemble_predictions.append(predictions)
            
        # get by tile rows, instead of by cnn rows    
        ensemble_predictions = np.array(ensemble_predictions).T
        
        # bincount number of labels predicted for each tile by ensemble. 
        # From bincounts derive label (argmax) and agreement (max)
        ensemble_counts = [np.bincount(x) for x in ensemble_predictions]
        ensemble_label = [np.argmax(x) for x in ensemble_counts]
        ensemble_agreement = [np.max(x) for x in ensemble_counts]
        
        self.ens_pred = ensemble_predictions
        self.ens_labe = ensemble_label
        self.ens_aggr = ensemble_agreement
        
        # add results to test df
        self.test_df["ensemble_prediction"] = ensemble_label
        self.test_df["ensemble_agreement"] = ensemble_agreement
        
        # Calculate ensemble accuracy and retained tiles for each agreement cutoff
        used_tiles = []
        agreement_levels = []
        agg_accs = []
        # Agreement minimum is 50% of CNNs (otherwise the other label would have been predicted)
        for i in range(int(np.ceil(self.number_cnn/2)),self.number_cnn+1):
            # All predictions with agreement i
            temp = self.test_df.loc[self.test_df["ensemble_agreement"]>=i]
            if temp.shape[0]>0:
                # Accuracy for agreement df 'temp'
                acc = (temp["ensemble_prediction"]==temp["label"]).sum() / temp.shape[0]
            else:
                # If no tiles in agreement level
                acc = "No Data Points"
            agreement_levels.append(i)
            agg_accs.append(acc)
            used_tiles.append(temp.shape[0])
            
        # save agreement accuracy, retained tiles to csv
        with open(self.test_results_out,"w") as f:
            f.write("ensemble_agreement,ensemble_accuracy,retained_tiles\n")
            for agg, acc, til in zip(agreement_levels,agg_accs,used_tiles):
                f.write(f"{agg},{acc},{til}\n")
        
        # reorder tiles DF and save to csv
        self.test_df = self.test_df[["patient id","label","ensemble_prediction","ensemble_agreement","path"]]
        self.test_df.to_csv(self.test_tiles_out)


class Flow_from_list(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(512,512), n_channels=3,
                 n_classes=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        # 
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __data_generation(self, list_IDs_temp, list_labels_temp):
        # Generates data batches of length batch_size
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load image and convert (add augments here if desired)
            img = np.asarray(Image.open(ID).convert("RGB"))
            
            X[i,] = img

            # store label
            y[i] = list_labels_temp[i]
        
        # to categorical
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __len__(self):
        # get number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        # generate one batch of data
        idx = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # get list of paths
        list_IDs_temp = [self.list_IDs[x] for x in idx]
        list_labels_temp = [self.labels[x] for x in idx]

        # generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        return X, y