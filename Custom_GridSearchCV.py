# Import important libraries
# If the desired model is not present in the import list, 
# import the model manually. 
import pandas as pd
import numpy as np
import itertools as it
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier 
from catboost import CatBoostClassifier
from sklearn.metrics import *
from tqdm import tqdm

class Custom_GridSearchCV():
    def __init__(self, estimator, param_grid={}, cv=5, class_weights=None):
        """
        Custom_GridSearchCV: A grid search that automatically returns
        various metrics for almost all sklearn and xgboost classification 
        models.
        estimator: A string of the model
        param_grid: parametric grid for grid search. All entries should be in 
        string format
        cv: Number of cross validation folds 
        """

        # Initialize cv
        self.cv = cv 
        # Initialize fitted, boolean if the grid of models have been fitted
        self.fitted = False       
        # Initialize models list
        self.models = []
        # Initialize cross validation evaluations 
        self.cross_eval = {}
        # Initialize cross validation evaluations 
        self.class_weights = class_weights
        # Extract parameters for baseline model. Parameters with only one value
        # are selected.
        base_params = [k+"="+v[0] for k,v in param_grid.items() if len(v)==1]
        base_params = ','.join(base_params)
        # Initialize baseline model 
        exec("self.baseline_model = "+estimator+"("+base_params+")")
        # Create a combinations of the parameter grid 
        all_params = sorted(param_grid)
        combinations = it.product(*(param_grid[name] for name in all_params))
        # Iterate through the combinations and keys to create a list of models
        keys = list(param_grid.keys())
        for j, comb in enumerate(combinations):
            params = ""
            for i, key in enumerate(keys):
                params += key+"="+comb[i]+","
            # Append models 
            exec("self.models.append("+estimator+"("+params[:-1]+"))")
        # Initialize predictions dataframe
        self.predictions = None
            
    def cross_validate(self, X, y, scoring="accuracy", SEED=42):
        """
        Cross validate data on training data and return dataset with the 
        largest scoring.
        X: Training dataset 
        y: Training labels 
        scoring: The type of scoring used in cross validation. Valid entries 
        are 'accuracy','precision','recall','f1' and 'roc_auc' 
        """

        # Set default scoring 
        self.scoring = scoring
        # Set class weights if None is passed to the constructor
        if self.class_weights==None:
            self.class_weights = compute_class_weight(class_weight = "balanced", classes = np.unique(y), y=y)
            self.class_weights = dict(zip(np.unique(y), self.class_weights))
        # Weighted boolean
        weighted = False
        # Cross validation: run custom_cross_validate with class weights, 
        # and over-sampled data with SMOTE, ADASYN and Random Over Sampler  
        if self.cross_eval=={}:
            cross_eval_dataset = ["class_weight","smote","adasyn","rand"]
            # Initialize interator with progress bar
            for sample in tqdm(cross_eval_dataset):
                # Check if the estimator has 'class_weight' attribute
                if sample=="class_weight":                    
                    # Clone the baseline model and assign class weight
                    bl_model = clone(self.baseline_model)
                    if hasattr(self.baseline_model, sample): 
                        bl_model.class_weight = self.class_weights
                        weighted = True
                    # XGBoost exceptions 
                    elif hasattr(self.baseline_model, 'scale_pos_weight'):
                        weight = self.class_weights[1]/self.class_weights[0]
                        bl_model.scale_pos_weight = weight
                        weighted = True  
                    # CatBoost exceptions 
                    elif hasattr(self.baseline_model, "class_weights"):
                        bl_model.class_weights = self.class_weights
                        weighted = True                        
                    if weighted: 
                        # Run custom cross validation 
                        train_dict, valid_dict = self.custom_cross_validate(bl_model,X,y, sample,cv=self.cv,SEED=SEED)
                        # Append values 
                        self.cross_eval[sample] = {'train':train_dict,
                                                  'validate':valid_dict}
                        
                else:
                    # Run custom cross validate on over-sampled data  
                    train_dict, valid_dict = self.custom_cross_validate(self.baseline_model, X, y, sample, cv=self.cv, SEED=SEED)
                    # Append values 
                    self.cross_eval[sample] = {'train':train_dict,
                                               'validate':valid_dict}
        # 'return_report' returns values from 'cross_eval' and 'scoring' 
        return_report = {}        
        for dataset in self.cross_eval.keys():
            return_report[dataset] = {}
            for report in self.cross_eval[dataset].keys():
                return_report[dataset][report] = self.cross_eval[dataset][report][scoring]                
        # 'report_dataframe': a dataframe where average values are selected
        report_dataframe = pd.DataFrame(return_report).applymap(np.mean) 
        # Transpose the dataframe so that indices represent 'cross_eval_dataset' 
        report_dataframe = report_dataframe.transpose()
        # Select the dataset with maximum score for training 
        max_train = report_dataframe.train.max()        
        max_train_score_dataset = report_dataframe.train[report_dataframe.train==max_train].index
        max_train_score_dataset = max_train_score_dataset[0]
        # Select the dataset with maximum score for validation 
        max_valid = report_dataframe.validate.max()        
        max_valid_score_dataset = report_dataframe.validate[report_dataframe.validate==max_valid].index
        max_valid_score_dataset = max_valid_score_dataset[0]
        # Combine scores
        max_score_dataset = {'train':max_train_score_dataset,
                            'validate':max_valid_score_dataset}
        # Return 'return_report' dictionary, 'report_dataframe' dataframe and 
        # 'max_score_dataset' dataset string
        return return_report, report_dataframe, max_score_dataset  
        
    def fit(self, X, y, class_weight=False):
        """
        Fit training data to a grid of models. 
        X: Training dataset 
        y: Training labels 
        """

        # Run if models are not fitted
        if not self.fitted:
            # Initialize progress bar for fitting
            for model in tqdm(self.models):
                # Set class_weight if defined
                if class_weight:
                    model.class_weight = self.class_weights
                # Iterate through each model and fit training data
                try: 
                    # Multiclass fitting
                    model.fit(X, y)
                except:
                    # Models that only support label encoding
                    idx = np.argmax(y, axis=1)
                    model.fit(X, idx)

            # Set 'fitted' to True     
            self.fitted = True            
            
    def predict(self, X, y, train_test='test'):
        """
        Predict values: Unlike predict functions for sklearn estimators, 
        this function takes y value as well, to report fitting metrics 
        X: Training/test dataset 
        y: Training/test labels 
        train_test: only 'train' and 'test' values need to be specified.
        """

        # Dataframe to store prediction per 'train_test'
        tr_ts = None
        # Iterate through each model             
        for model in tqdm(self.models):
            # Calculate the probabilities of predictions 
            prob_preds = model.predict_proba(X)
            # Convert prediction probablities to numpy array 
            prob_preds = np.array(prob_preds)
            # Select scores for roc_curve calculation
            if len(prob_preds.shape) == 3:
                # Multiclass one-hot encoded    
                prob_preds = prob_preds[:,:,1].T                
                score = np.squeeze(prob_preds)
                preds = np.argmax(score, axis=1)
            elif prob_preds.shape[-1]>2:
                # Multiclass label encoded
                score = np.squeeze(prob_preds)
                preds = np.argmax(score, axis=1)    
            else:
                # Two-class prediction
                score = prob_preds[:, 1]
                preds = (score>0.5)*1            
            # Find fpr and tpr values
            # fpr, tpr, threshold = roc_curve(y, score)
            # Dict to store values 
            tmp = {}
            # Populate 'tmp_df' with values
            tmp['train_test'] = train_test
            tmp['preds'] = [preds]
            tmp['prob_preds'] = [prob_preds]
            tmp['log_loss_score'] = log_loss(y, score)
            tmp['accuracy'] = accuracy_score(y, preds)
            tmp['precision'] = precision_score(y, preds, average='weighted')
            tmp['recall'] = recall_score(y, preds, average='weighted')
            tmp['f1'] = f1_score(y, preds, average='weighted')
            # tmp['fpr'] = [fpr]
            # tmp['tpr'] = [tpr]
            # tmp['auc'] = auc(fpr, tpr)
            # tmp['roc_auc'] = roc_auc_score(y, preds)
            # Create a temp dataframe with 'predictions' columns
            tmp_df = pd.DataFrame(tmp)                
            # Concatenate 'tmp_df' and 'tmp_df'
            if not isinstance(tr_ts, pd.DataFrame): 
                tr_ts = tmp_df.copy()
            else:
                tr_ts = pd.concat([tr_ts,tmp_df])
        # Reset indices 
        tr_ts.index = np.arange(len(self.models))
        # Assign values to 'predictions'
        if not isinstance(self.predictions, pd.DataFrame):
            self.predictions = tr_ts.copy()
        else:
            self.predictions = pd.concat([self.predictions,tr_ts])
    
    def best_model(self, metrics=['accuracy']):
        """
        This function returns the best model and model metrics 
        based on provided metrics from the test dataset.
        metrics: A list of metrics
        """

        # Select test cases only 
        test_metrics = self.predictions.loc[self.predictions.train_test=='test']
        # Sort test_metrics by metrics in descending order          
        test_metrics = test_metrics.sort_values(by=metrics, ascending=False)
        # Reset indices  
        test_metrics = test_metrics.reset_index()
        # Best model index 
        idx = test_metrics.loc[0,:]['index']
        # Return best model and best model metrics
        return self.models[idx], test_metrics.loc[0,:]

    def custom_cross_validate(self, estimator, X, y, over_sample, SEED, cv=5):
        """
        'custom_cross_validate' function that performs oversampling and cross 
        validate results.
        X: Training dataset 
        y: Training labels
        over_sample: Type of oversampling method used. Acceptable values are 
        'class_weight', 'smote', 'adasyn' and 'rand'
        cv: The number of cross-validation folds 
        SEED: Random number seed 
        """

        # Create dictionaries to hold the scores from each fold
        train_dict = {'log_loss_score':np.ndarray(cv), 'precision':np.ndarray(cv), 
                     'accuracy':np.ndarray(cv), 'recall':np.ndarray(cv), 
                      'f1':np.ndarray(cv)
                    }
        valid_dict = {'log_loss_score':np.ndarray(cv), 'precision':np.ndarray(cv), 
                     'accuracy':np.ndarray(cv), 'recall':np.ndarray(cv), 
                      'f1':np.ndarray(cv)}
    
        # Instantiate a splitter object and loop over its result
        kfold = StratifiedKFold(n_splits=cv, shuffle=True)
        for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
            # Extract train and validation subsets using the provided indices
            X_t, X_val = X.iloc[train_index], X.iloc[val_index]
            y_t, y_val = y.iloc[train_index], y.iloc[val_index]
        
            # Instantiate StandardScaler
            scaler = StandardScaler()
            # Fit and transform X_t
            X_t_scaled = scaler.fit_transform(X_t)
            # Transform X_val
            X_val_scaled = scaler.transform(X_val)
        
            # If over_sample is define, choose the method passed
            if over_sample == "class_weight":
                X_t_oversampled, y_t_oversampled = (X_t_scaled, y_t)            
            else:        
                if over_sample == "smote":
                    # Instantiate SMOTE 
                    sampler = SMOTE(random_state=SEED)
                elif over_sample == "adasyn":
                    # Instantiate ADASYN 
                    sampler = ADASYN(random_state=SEED)
                elif over_sample == "rand":
                    # Instantiate RandomOverSampler
                    sampler = RandomOverSampler(random_state=SEED)
                # Fit and transform X_t_scaled and y_t using sm
                X_t_oversampled, y_t_oversampled = sampler.fit_resample(X_t_scaled, y_t)
        
            # Clone the provided model and fit it on the train subset
            temp_model = clone(estimator)

            # Fit the model 
            try:
                # Try fitting one-hot encoding                
                temp_model.fit(X_t_oversampled, y_t_oversampled)
            except:
                # Or else, do label encoding
                idx = np.argmax(y_t_oversampled, axis=1)
                temp_model.fit(X_t_oversampled, idx)
        
            # Calculate probablities 
            train_probs = temp_model.predict_proba(X_t_oversampled)
            val_probs = temp_model.predict_proba(X_val_scaled)

            # Convert prediction probablities to numpy array 
            train_probs = np.array(train_probs)
            val_probs = np.array(val_probs)

            # Select scores for roc_curve calculation
            if len(train_probs.shape) == 3:
                # Multiclass one-hot encoded    
                train_probs = train_probs[:,:,1].T
                val_probs = val_probs[:,:,1].T

                train_score = np.squeeze(train_probs)
                val_score = np.squeeze(val_probs)

                # Find predictions
                train_pred = np.argmax(train_score, axis=1)
                val_pred = np.argmax(val_score, axis=1)                

            elif train_probs.shape[-1]>2:
                # Multiclass label encoded
                train_score = np.squeeze(train_probs)
                val_score = np.squeeze(val_probs)
                
                # Find predictions
                train_pred = np.argmax(train_score, axis=1)
                val_pred = np.argmax(val_score, axis=1)                

            else:
                # Two-class prediction
                train_score = train_probs[:, 1]
                val_score = val_probs[:, 1]

                # Find predictions 
                train_pred = (train_score>0.5)*1
                val_pred = (val_score>0.5)*1

            # Evaluate the provided model on the train and validation subsets
            # Log loss score 
            train_dict['log_loss_score'][fold] = log_loss(y_t_oversampled, train_score)
            valid_dict['log_loss_score'][fold] = log_loss(y_val, val_score)
            # Accuracy score 
            train_dict['accuracy'][fold] = accuracy_score(y_t_oversampled, train_pred)
            valid_dict['accuracy'][fold] = accuracy_score(y_val, val_pred)
            # Precision score 
            train_dict['precision'][fold] = precision_score(y_t_oversampled, train_pred, average='weighted')
            valid_dict['precision'][fold] = precision_score(y_val, val_pred, average='weighted')
            # Recall score
            train_dict['recall'][fold] = recall_score(y_t_oversampled, train_pred, average='weighted')
            valid_dict['recall'][fold] = recall_score(y_val, val_pred, average='weighted')
            # F1 score 
            train_dict['f1'][fold] = f1_score(y_t_oversampled, train_pred, average='weighted')
            valid_dict['f1'][fold] = f1_score(y_val, val_pred, average='weighted')
            # FPR and TPR 
            # train_fpr, train_tpr, threshold = roc_curve(y_t_oversampled, train_score)
            # valid_fpr, valid_tpr, threshold = roc_curve(y_val, val_score)
            # train_dict['fpr'].append(train_fpr)
            # train_dict['tpr'].append(train_tpr)
            # valid_dict['fpr'].append(valid_fpr)
            # valid_dict['tpr'].append(valid_tpr)

            # AUC
            # train_dict['auc'][fold] = auc(train_fpr, train_tpr)
            # valid_dict['auc'][fold] = auc(valid_fpr, valid_tpr)
            # ROC_AUC 
            # train_dict['roc_auc'][fold] = roc_auc_score(y_t_oversampled, train_pred)
            # valid_dict['roc_auc'][fold] = roc_auc_score(y_val, val_pred)
        
        # Return training and validation results
        return train_dict, valid_dict