import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold

from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def data_analysis(df: pd.DataFrame):
    
    print(df.info())
    print(df.describe())
    
    print("Il dataset contiene:", df.shape[0], "istanze")
    print("Numero di features:", df.shape[1])
    print("Lista features:\n", df.columns)

    print(df.isnull().sum())   
    #All'interno del dataset non sono presenti valori nulli

    # Controllo che il dataset sia bilanciato rispetto agli anni presi (2011 e 2012)
    print(df.groupby("yr").size())

    # Controllo che il dataset sia bilanciato rispetto alle condizioni metereologiche
    print(df.groupby("weathersit").size())



def data_visualization(df: pd.DataFrame, group_by, legend_labels, feature, title):
    
    df.groupby(group_by)['cnt'].mean()
    
    fig,ax=plt.subplots(figsize=(12,7))
    sea.barplot(x='mnth',y='cnt',data=df[['mnth','cnt',feature]],hue=feature)

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=[legend_labels[val] for val in df[feature].unique()])

    new_xticklabels = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile','Maggio', 'Giugno', 'Luglio', 'Agosto', 'Settembre', 'Ottobre', 'Novembre', 'Dicembre']  
    ax.set_xticklabels(new_xticklabels)

    plt.title(title)
    plt.xlabel('Mese')
    plt.ylabel('Numero Biciclette')
    plt.show()



def feature_selection(df: pd.DataFrame):
    #Scarto features indice e giorno
    df=df.drop(['instant','dteday'], axis = 1)

    #Rappresento matrice di correlazione
    corr_matrix = df.corr()
    plt.figure(figsize = (12,7))
    sea.heatmap(corr_matrix, annot=True)
    plt.show()
    
    #Individuo Features e Target
    features = ['season', 'yr', 'mnth', 'holiday', 'weekday','workingday', 'weathersit', 'temp', 'hum', 'windspeed']
    X = df[features].values
    y = df['cnt']
    
    #Suddivido il dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3, test_size=1/3,random_state=42)
    print('Train', X_train.shape, y_train.shape)
    print('Test', X_test.shape, y_test.shape)
    
    # Standardizzo i dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
  
    #Utilizzo Lasso Regression per attribuire peso alle features
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train_scaled, y_train)

    sfm = SelectFromModel(lasso_model)
    X_train_selected = sfm.transform(X_train_scaled)
    X_test_selected = sfm.transform(X_test_scaled)

    #Rappresento i pesi assegnati alle features
    feature_importances = lasso_model.coef_
    for i, importance in enumerate(feature_importances):
        print(f'Feature {i}: {importance}')
    print()
    
    sea.barplot(x=[i for i in range(len(feature_importances))], y=feature_importances)
    plt.title('Pesi Features')
    plt.xlabel('Feature')
    plt.ylabel('Peso')
    plt.show()
    
    return X_train_selected, X_test_selected, y_train, y_test



def GridSearch_analysis(X_train_selected, y_train, knn_model, tree_model, svr_model):
    #K-Nearest Neighbors
    #Analisi GridSearch
    gridsearch_parameters_knn = {'n_neighbors': list(range(1, 10)), 
                            'weights': ['uniform', 'distance']}
    kf = KFold(n_splits=5, shuffle=True)

    Grid_Search_knn = GridSearchCV(knn_model, gridsearch_parameters_knn, cv=kf)
    Grid_Search_knn.fit(X_train_selected, y_train)

    #Estrazione dei migliori iperparametri
    best_n_neighbors = Grid_Search_knn.best_params_['n_neighbors']
    best_weights = Grid_Search_knn.best_params_['weights']

    print("Best Estimator: ", Grid_Search_knn.best_estimator_)
    print("Best Number of Neighbors:", best_n_neighbors)
    print("Best Weights:", best_weights)
    print()

    #Modello con i migliori iperparametri
    knn_model = KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights)
    
      
    #Decision Tree
    gridsearch_parameters_tree = {'max_depth': list(range(1, 20)), 'min_samples_split': [10, 13, 15], 'min_samples_leaf': list(range(1, 5))}
    kf=KFold(n_splits=5,shuffle=True)

    Grid_Search_dt = GridSearchCV(tree_model, gridsearch_parameters_tree, cv=kf)
    Grid_Search_dt.fit(X_train_selected, y_train)

    best_max_depth = Grid_Search_dt.best_params_['max_depth']
    best_min_samples_split=Grid_Search_dt.best_params_['min_samples_split']
    best_min_samples_leaf=Grid_Search_dt.best_params_['min_samples_leaf']

    print("Best Estimator: ", Grid_Search_dt.best_estimator_)
    print("Best Max Depth:", best_max_depth)
    print("Min Samples Split:",best_min_samples_split)
    print("Min Samples Leaf:",best_min_samples_leaf)
    print()
    
    #Modello con i migliori iperparametri
    tree_model = DecisionTreeRegressor(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split)
    
       
    #Support Vector Regressor
    gridsearch_parameters_svr = {'kernel': ['linear', 'rbf', 'poly'], 'C': [500,1000,2000], 'epsilon': [0.1, 0.3, 0.5]}
    kf=KFold(n_splits=5,shuffle=True)

    Grid_Search_svr = GridSearchCV(svr_model, gridsearch_parameters_svr, cv=kf)
    Grid_Search_svr.fit(X_train_selected, y_train)

    best_kernel = Grid_Search_svr.best_params_['kernel']
    best_C=Grid_Search_svr.best_params_['C']
    best_epsilon=Grid_Search_svr.best_params_['epsilon']

    print("Best Estimator: ", Grid_Search_svr.best_estimator_)
    print("Best Kernel:", best_kernel)
    print("Min C:",best_C)
    print("Min Epsilon:",best_epsilon)
    print()

    #Modello con i migliori iperparametri
    svr_model = SVR(kernel=best_kernel, C=best_C, epsilon=best_epsilon)
    
       
    models=[LinearRegression(),
        KNeighborsRegressor(n_neighbors=best_n_neighbors, weights=best_weights),
        DecisionTreeRegressor(max_depth=best_max_depth, min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf),
        SVR(kernel=best_kernel, C=best_C, epsilon=best_epsilon)]
        
    return knn_model, tree_model, svr_model, models



def model_select(model,X_train_selected, y_train):
    #K-Fold
    kfold=model_selection.KFold(n_splits=5,shuffle=True)
    
    #Valuto modello
    predict=model_selection.cross_val_score(model , X_train_selected , y_train , cv=kfold , scoring='neg_mean_absolute_error')
    cv_score=predict.mean()
    
    print('Model:',model)
    print('Mean Absolute Error (MAE):', abs(cv_score))
    print()

    
    
def train_model(model, X_train_selected, y_train):
    model.fit(X_train_selected, y_train) 
   
   
    
def test_model(model, model_name, X_test_selected, y_test):
    y_pred = model.predict(X_test_selected)

    print('Valutazione utilizzando le metriche di sklearn:', model_name)
    print("Errore Test (MSE) = ", mean_squared_error(y_test, y_pred))
    print("Errore Test (MAE) = ", mean_absolute_error(y_test, y_pred))
    print("Errore Test (RMSE) = ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Errore Test (R-squared) = ", r2_score(y_test, y_pred))
    print()

    plt.scatter(y_test, y_pred)
    plt.xlabel('Valore Reale')
    plt.ylabel('Predizione')
    plt.title(model_name)
    plt.show()
    


def ensemble_stacking_regressor(lr_model, knn_model, tree_model, svr_model):
    
    kf=KFold(n_splits=5,shuffle=True)
    
    final_model = StackingRegressor(
        estimators=[('lr',lr_model), ('knn', knn_model), ('tree', tree_model), ('svr',svr_model)],
        final_estimator=LinearRegression(),
        cv=kf  
    )
    
    return final_model

    

if __name__ == '__main__':
    
    #Dataset    
    df = pd.read_csv("day.csv", delimiter=',')
    
    #Analisi Dati
    data_analysis(df)
    
 ############################# VISUALIZZAZIONE DATI #########################   
 
    # Rappresento il noleggio delle bici in base al mese e all’anno.
    group_by= ['mnth', 'yr']
    legend_labels = {0: '2011', 1: '2012'}
    feature = 'yr'
    title='Conteggio per Mese e Anno'
    data_visualization(df, group_by, legend_labels, feature, title)
    
    # Rappresento il noleggio in base al tipo di giorno (lavorativo e non lavorativo) e ai mesi.
    group_by= ['workingday', 'mnth','weathersit']
    legend_labels = {1: 'Lavorativo', 0: 'Non Lavorativo'}
    feature = 'workingday'
    title='Conteggio per Tipologia di Giorni'
    data_visualization(df, group_by, legend_labels, feature, title)

    #Rappresento il noleggio delle bici in base alle diverse condizioni metereologiche 
    group_by= ['weathersit']
    legend_labels = {1: 'Sereno', 2: 'Nuvoloso', 3: 'Piovoso'}
    feature = 'weathersit'
    title='Conteggio per Tempo Meterologico'
    data_visualization(df, group_by, legend_labels, feature, title)
    
    
    ############################# PRE PROCESSING #########################
    
    X_train_selected, X_test_selected, y_train, y_test = feature_selection(df)
    
    ############################# MODELS #########################
    
    #Definizione dei modelli
    #Linear Regression
    lr_model = LinearRegression()

    #K-Nearest Neighbors
    knn_model = KNeighborsRegressor()

    #Decision Tree
    tree_model = DecisionTreeRegressor()

    #Support Vector Regressor
    svr_model = SVR()
    
    models_name=['Linear Regression',
        'K-NeighborsNeighbors',
        'Decision Tree Regressor',
        'Support Vector Regressor']
    
    #Grid search per ricerca degli iperparametri
    knn_model,tree_model,svr_model,models= GridSearch_analysis(X_train_selected, y_train, knn_model, tree_model, svr_model)
    
    #Model Selection  
    for model in models:
        model_select(model,X_train_selected, y_train)
        
    
    ############################# TRAINING MODELS #########################
    
    #Training
    for model in models:
        train_model(model,X_train_selected, y_train)
    
    ############################# TESTING MODELS #########################

    #Testing
    for i, model in enumerate(models):
        test_model(model, models_name[i], X_test_selected, y_test)
    
        
############################# Scelta finale del modello #############################

    # Creo Modello Finale (Stacking Regressor)    
    final_model=ensemble_stacking_regressor(lr_model, knn_model, tree_model, svr_model)

    #Training Stacking Regressor
    final_model.fit(X_train_selected, y_train)
    y_pred_tr=final_model.predict(X_train_selected)

    #Testing
    y_pred_tst = final_model.predict(X_test_selected)

    print('Evaluation using the sklearn metrics Stacking Regressor')
    print("Errore Train (MSE) = ", mean_squared_error(y_train, y_pred_tr))
    print("Errore Test (MSE) = ", mean_squared_error(y_test, y_pred_tst))
    print("Errore Train (MAE) = ", mean_absolute_error(y_train, y_pred_tr))
    print("Errore Test (MAE) = ", mean_absolute_error(y_test, y_pred_tst))
    print("Errore Train (RMSE) = ", np.sqrt(mean_squared_error(y_train, y_pred_tr)))
    print("Errore Test (RMSE) = ", np.sqrt(mean_squared_error(y_test, y_pred_tst)))
    print("Errore Train (R-squared) = ", r2_score(y_train, y_pred_tr))
    print("Errore Test (R-squared) = ", r2_score(y_test, y_pred_tst))
    
    plt.scatter(y_test, y_pred_tst)
    plt.xlabel('Valore Reale')
    plt.ylabel('Predizione')
    plt.title('Stacking Regressor')
    plt.show()
    

   