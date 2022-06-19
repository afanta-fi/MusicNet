import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix

def prepare_metrics(model_grid, train_val='train'):
    """
    This function plots metrics on different datasets
    model_grid: Custom_GridSearchCV object
    Run this code after fitting. 
    """    
    # Convert cross_eval from Custom_GrisSearchCV into dataframe
    # Table is transposed to set dataset as columns
    tmp = pd.DataFrame(model_grid.cross_eval).transpose()
    tmp = pd.DataFrame(tmp[train_val].to_dict())
    # Initialize metrics dataframe 
    metrics = pd.DataFrame(columns=list(tmp.columns)+['metric'])
    # Iterate through each metric in 'tmp' and assign to metrics dataframe
    for idx in tmp.index:
        # Create an empty dataframe 
        metric = pd.DataFrame(tmp.loc[idx,:].to_dict())
        # Update type of metric 
        metric['metric'] = idx
        # Append results
        metrics = pd.concat([metrics, metric])
    # Reset metrics
    metrics = metrics.reset_index()
    # Rename 'index' column to 'fold'
    metrics.rename(columns={'index':'fold'}, inplace=True)
    # Initialiaze squeezed metrics dataframe
    squeezed_metrics = pd.DataFrame(columns=['fold','metric','dataset',
                                             'value','train_val'])
    # Iterate through each column and populate 'squeezed_metrics'
    for col in model_grid.cross_eval.keys():
        # Create empty dataframe 
        sq_met = pd.DataFrame(columns=squeezed_metrics.columns)
        # Populate 'sq_met'
        sq_met.fold = metrics.fold
        sq_met.metric = metrics.metric
        sq_met.dataset = col
        sq_met.value = metrics[col]
        sq_met.train_val = train_val
        # Append 'sq_met' to 'squeezed_metrics'
        squeezed_metrics = pd.concat([squeezed_metrics, sq_met], 
                                     ignore_index=True)
    # Set log-loss to a variable and remove the column from 'squeezed_metrics'
    log_loss_vals =  squeezed_metrics.loc[squeezed_metrics.metric =='log_loss_score']
    squeezed_metrics.drop(log_loss_vals.index, inplace=True)
    # Get fpr and tpr from 'squeezed_metrics'
    # fpr = squeezed_metrics.loc[squeezed_metrics.metric=='fpr']
    # tpr = squeezed_metrics.loc[squeezed_metrics.metric=='tpr']
    # # Drop 'fpr' and 'tpr' from 'squeezed_metrics'
    # squeezed_metrics.drop(fpr.index, inplace=True)
    # squeezed_metrics.drop(tpr.index, inplace=True)
    # # Find the longest fpr/tpr
    # roc_max_len = fpr.value.apply(lambda x: len(x)).max()
    # # Reset indices for 'fpr' and 'tpr'
    # fpr = fpr.reset_index()
    # tpr = tpr.reset_index()
    # # Initialize 'fpr_mat' and 'tpr_mat' 
    # fpr_mat = np.zeros((len(fpr), roc_max_len))
    # tpr_mat = np.zeros((len(tpr), roc_max_len))
    # # Iteratre through each fpr and tpr, and interpolate values
    # for i in range(len(fpr)):
    #     # Create a uniformly spaced 'fpr' values
    #     xvals = np.linspace(0, 1, roc_max_len)
    #     # Interpolate y values 
    #     yinterp = np.interp(xvals, fpr.loc[i,'value'], tpr.loc[i,'value'])
    #     # Update fpr and tpr matrix 
    #     fpr_mat[i,:] = xvals
    #     tpr_mat[i,:] = yinterp
    # # Create roc_vals dictionary 
    # roc_vals = {}
    # # Instead of taking the entire matrices, the mean and std values are 
    # # selected for 'fpr' and 'tpr'
    # roc_vals['fpr_mean'] = fpr_mat.mean(axis=0)
    # roc_vals['tpr_mean'] = tpr_mat.mean(axis=0)
    # roc_vals['fpr_std'] = fpr_mat.std(axis=0)
    # roc_vals['tpr_std'] = tpr_mat.std(axis=0)
    # Return squeezed metrics, log loss and roc 
    return squeezed_metrics, log_loss_vals


def plot_metrics(model_grid):
    """
    This function plots metrics on different datasets
    model_grid: Custom_GridSearchCV object
    Run this code after fitting. 
    """    

    # Get parameters for training 
    tr_metrics, tr_log_loss = prepare_metrics(model_grid, 'train')
    # Get parameters for validation
    vl_metrics, vl_log_loss = prepare_metrics(model_grid, 'validate')
    # Combine log losses for training and validation 
    log_loss_combined = pd.concat([tr_log_loss,vl_log_loss])
    # Create figure to plot log loss and ROC curve 
    fig, axes = plt.subplots(3 , 1, figsize=(15,18))
    # Log loss for training and validation 
    g1 = sns.boxplot(x="dataset", y="value", hue="train_val",
                     data=log_loss_combined, ax=axes[0])
    # Format labels and title 
    g1.set_xlabel('')
    g1.set_ylabel('Log Loss',fontsize=13)
    g1.set_title('Log Loss vs Dataset',fontsize=15)
    g1.legend(title='Fold')
    
    # # Plot mean ROC curve for training fold
    # g2 = sns.lineplot(x=tr_roc_vals['fpr_mean'], y=tr_roc_vals['tpr_mean'], 
    #               label='train', ax=axes[1], color='tab:blue')
    # # Plot the standard deviation for training ROC 
    # plt.fill_between(tr_roc_vals['fpr_mean'], 
    #                  tr_roc_vals['tpr_mean'] - tr_roc_vals['tpr_std'],
    #                  tr_roc_vals['tpr_mean'] + tr_roc_vals['tpr_std'],
    #                  color='tab:blue', alpha=0.2)
    # # Plot mean ROC curve for validation fold
    # sns.lineplot(x=vl_roc_vals['fpr_mean'], y=vl_roc_vals['tpr_mean'], 
    #               label='validate', ax=axes[1], color='tab:orange')
    # # Plot the standard deviation for validation ROC
    # plt.fill_between(vl_roc_vals['fpr_mean'], 
    #                  vl_roc_vals['tpr_mean'] - vl_roc_vals['tpr_std'],
    #                  vl_roc_vals['tpr_mean'] + vl_roc_vals['tpr_std'],
    #                  color='tab:orange', alpha=0.2)
    
    # # Format labels and title 
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # g2.set_xlabel('False Positve Rate', fontsize=13)
    # g2.legend(title='Fold')
    # g2.set_ylabel('True Positve Rate', fontsize=13)
    # g2.set_title('ROC Curve',fontsize=15)

    # Plot metrics for training fold
    # fig, axes = plt.subplots(2 , 1, figsize=(15,12))
    g3 = sns.boxplot(x="dataset", y="value", hue="metric",
                     data=tr_metrics, ax=axes[1])
    # Format labels and title
    g3.set_xlabel('')
    g3.set_ylabel('Score', fontsize=13)
    g3.set_title('Training Scores vs Dataset',fontsize=15)
    g3.legend(title='Metric')
    
    # Plot metrics for validation fold
    g4 = sns.boxplot(x="dataset", y="value", hue="metric",
                     data=vl_metrics, ax=axes[2])
    # Format labels and title
    g4.set_xlabel('')
    g4.set_ylabel('Score',fontsize=13)
    g4.set_title('Validation Scores vs Dataset',fontsize=15)
    g4.legend(title='Metric')


def prepare_predictions(model_grid,train_test='test'):
    """
    This function plots metrics on train/test predictions 
    model_grid: Custom_GridSearchCV object
    Run this code after fitting. 
    """    

    # Convert prediction from Custom_GrisSearchCV into dataframe    
    # Select train or test
    tmp = model_grid.predictions.loc[model_grid.predictions.train_test==train_test]
    # Initialize squeezed predictions     
    squeezed_preds = pd.DataFrame(columns=['metric','value','train_test'])
    # Find metrics from 'cross_eval' 
    columns = model_grid.cross_eval['smote']['train'].keys()
    # Iterate through 'predictions' columns 
    for col in columns:
        # Initialize a dummy dataframe for each column
        sq_preds = pd.DataFrame(columns=squeezed_preds.columns)
        # Assign values 
        sq_preds.value = tmp[col]
        sq_preds.train_test = tmp['train_test']
        sq_preds.metric = col
        # Append values to 'squeezed_preds'
        squeezed_preds = pd.concat([squeezed_preds,sq_preds])
    # Reset index
    squeezed_preds = squeezed_preds.reset_index()
    # Get log loss from 'squeezed_preds'    
    log_loss_vals = squeezed_preds.loc[squeezed_preds.metric=='log_loss_score']
    squeezed_preds.drop(log_loss_vals.index, inplace=True)
    # Get fpr and tpr from 'squeezed_preds'
    # fpr = squeezed_preds.loc[squeezed_preds.metric=='fpr']
    # tpr = squeezed_preds.loc[squeezed_preds.metric=='tpr']
    # # Drop 'fpr' and 'tpr' from 'squeezed_preds'
    # squeezed_preds.drop(fpr.index, inplace=True)
    # squeezed_preds.drop(tpr.index, inplace=True)
    # # Find the longest fpr/tpr
    # roc_max_len = fpr.value.apply(lambda x: len(x)).max()
    # # Reset indices for 'fpr' and 'tpr'
    # fpr = fpr.reset_index()
    # tpr = tpr.reset_index()
    # # Initialize 'fpr_mat' and 'tpr_mat' 
    # fpr_mat = np.zeros((len(fpr), roc_max_len))
    # tpr_mat = np.zeros((len(tpr), roc_max_len))
    # # Iteratre through each fpr and tpr, and interpolate values
    # for i in range(len(fpr)):
    #     # Create a uniformly spaced 'fpr' values
    #     xvals = np.linspace(0, 1, roc_max_len)
    #     # Interpolate y values 
    #     yinterp = np.interp(xvals, fpr.loc[i,'value'], tpr.loc[i,'value'])
    #     # Update fpr and tpr matrix 
    #     fpr_mat[i,:] = xvals
    #     tpr_mat[i,:] = yinterp
    # # Create roc_vals dictionary 
    # roc_vals = {}
    # # Instead of taking the entire matrices, the mean and std values are 
    # # selected for 'fpr' and 'tpr'
    # roc_vals['fpr_mean'] = fpr_mat.mean(axis=0)
    # roc_vals['tpr_mean'] = tpr_mat.mean(axis=0)
    # roc_vals['fpr_std'] = fpr_mat.std(axis=0)
    # roc_vals['tpr_std'] = tpr_mat.std(axis=0)
    # Return squeezed metrics, log loss and roc 
    return squeezed_preds, log_loss_vals

def plot_predictions(model_grid):
    """
    This function plots predictions on different datasets
    model_grid: Custom_GridSearchCV object
    Run this code after fitting. 
    """    

    # Get parameters for training 
    tr_preds, tr_log_loss = prepare_predictions(model_grid, 'train')
    # Get parameters for validation
    ts_preds, ts_log_loss = prepare_predictions(model_grid, 'test')
    # Combine log losses for training and validation 
    log_loss_combined = pd.concat([tr_log_loss,ts_log_loss])
    # Create figure to plot log loss and ROC curve 
    fig, axes = plt.subplots(2 , 1, figsize=(15,12))
    # Log loss for training and validation 
    g1 = sns.boxplot(x="metric", y="value",hue="train_test",
                     data=log_loss_combined, ax=axes[0])
    # Format labels and title 
    g1.set_xlabel('')
    g1.set_ylabel('Log Loss',fontsize=13)
    g1.set_title('Log Loss vs Dataset',fontsize=15)
    g1.legend(title='Dataset')
    
    # # Plot mean ROC curve for training fold
    # g2 = sns.lineplot(x=tr_roc_vals['fpr_mean'], y=tr_roc_vals['tpr_mean'], 
    #               label='train', ax=axes[1], color='tab:blue')
    # # Plot the standard deviation for training ROC 
    # plt.fill_between(tr_roc_vals['fpr_mean'], 
    #                  tr_roc_vals['tpr_mean'] - tr_roc_vals['tpr_std'],
    #                  tr_roc_vals['tpr_mean'] + tr_roc_vals['tpr_std'],
    #                  color='tab:blue', alpha=0.2)
    # # Plot mean ROC curve for validation fold
    # sns.lineplot(x=ts_roc_vals['fpr_mean'], y=ts_roc_vals['tpr_mean'], 
    #               label='test', ax=axes[1], color='tab:orange')
    # # Plot the standard deviation for validation ROC
    # plt.fill_between(ts_roc_vals['fpr_mean'], 
    #                  ts_roc_vals['tpr_mean'] - ts_roc_vals['tpr_std'],
    #                  ts_roc_vals['tpr_mean'] + ts_roc_vals['tpr_std'],
    #                  color='tab:orange', alpha=0.2)
    
    # # Format labels and title 
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # g2.set_xlabel('False Positve Rate', fontsize=13)
    # g2.legend(title='Dataset')
    # g2.set_ylabel('True Positve Rate', fontsize=13)
    # g2.set_title('ROC Curve',fontsize=15)

    # Plot metrics for training data
    preds = pd.concat([tr_preds,ts_preds])
    # fig, ax = plt.subplots(figsize=(15,6))
    g3 = sns.boxplot(x="train_test", y="value", hue="metric",
                     data=preds, ax=axes[1])
    # Format labels and title
    g3.set_xlabel('')
    g3.set_ylabel('Score', fontsize=13)
    g3.set_title('Scores vs Training and Test Datasets',fontsize=15)
    g3.legend(title='Metric')


def plot_confusion_matrix(y_true, y_pred, normalize=None, class_names=None):
    '''
    Simplified confusion matrix plotter 
    y_true: True y values 
    y_pred: Predicted y values 
    normalize: 'normalize' input used for confusion_matrix
    '''

    # Generate a square plot
    fig, ax = plt.subplots(figsize=(5,4))
    
    # Create a confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    # Plot heatmap  
    sns.heatmap(cm, annot=True, cmap='viridis', linecolor='black', 
                linewidth=1, ax=ax)
    # Apply labels and limits
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    lim = np.max(y_true)+1
    plt.xlim(lim,0)
    plt.ylim(0,lim)
    # Assign class names if specified 
    if class_names!=None:
        ax.set_xticklabels([class_names[int(i.get_text())] for i in ax.get_xticklabels()])
        ax.set_yticklabels([class_names[int(i.get_text())] for i in ax.get_yticklabels()], 
            verticalalignment='center')

def plot_correlation(df, features, cmap="viridis"):
    """
    Plot the Pearson correlation of a dataframe for given features

    df: Input dataframe 
    features: Highlighted features for correlation

    Returns:
    Lower trianglular pearosn correlation plot 
    """

    # Change seaborn style 
    sns.set_style('white')

    # Select features for correlation analysis    
    df = df[features]

    # Compute correlation
    corr = df.corr()

    # Create a triangular mask to show the lower triangular section of the heatmap
    mask = np.zeros_like(corr, dtype=np.bool) 
    mask[np.triu_indices_from(mask)] = True 

    # Initialize plot
    fig, ax = plt.subplots(figsize=(20, 16))
    plt.title('Pearson Correlation Matrix',fontsize=17)

    # Plot the Pearson Correlation Matrix for the numeric columns
    sns.heatmap(corr,linewidths=0.25,vmax=0.7,square=True,
            cmap=cmap,linecolor='w',annot=True,annot_kws={"size":9},
            mask=mask,cbar_kws={"shrink": .8},ax=ax);

    # Adjust font size of ticks 
    ax.set_xticklabels(ax.get_xmajorticklabels(),fontsize=12);
    ax.set_yticklabels(ax.get_ymajorticklabels(),fontsize=12);
    return corr
