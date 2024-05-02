from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def regresion_logistica(X_train, X_test, y_train, y_test, op):
    modelo = LogisticRegression(max_iter=10000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    #Dataset 2
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)
        #Calculo de la matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    #Dataset 3
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("\nRegresión Logística:")
    print(f"Precisión: {precision}")
    print(f"Recuperación: {recall}")
    print(f"Especificidad: {specificity}")
    print(f"Puntuación F1: {f1}")

def k_vecinos_cercanos(X_train, X_test, y_train, y_test, op, n_neighbors=3):
    modelo = KNeighborsClassifier(n_neighbors=n_neighbors)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("\nK-Vecinos Cercanos: ")
    print(f"Precisión: {precision}")
    print(f"Recuperación: {recall}")
    print(f"Especificidad: {specificity}")
    print(f"Puntuación F1: {f1}")

def maquinas_vector_soporte(X_train, X_test, y_train, y_test, op):
    modelo = SVC()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("\nMáquinas Vector Soporte: ")
    print(f"Precisión: {precision}")
    print(f"Recuperación: {recall}")
    print(f"Especificidad: {specificity}")
    print(f"Puntuación F1: {f1}")

def naive_Bayes(X_train, X_test, y_train, y_test, op):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred)    
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    elif op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("\nNaive Bayes:")
    print(f"Precisión: {precision}")
    print(f"Recuperación: {recall}")
    print(f"Especificidad: {specificity}")
    print(f"Puntuación F1: {f1}")

def AutoInsurSweden():
    dataset = pd.read_csv('AutoInsurSweden.csv')
    #Número de reclamos
    X = dataset[['X']]
    #Pago total por los reclamos
    y = dataset['Y'] 
    #Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def pima_indians_diabetes():
    dataset = pd.read_csv('pima-indians-diabetes.csv', sep=",")
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def wine_Quality():
    dataset = pd.read_csv('wine-Quality.csv', sep=",")
    dataset['quality_label'] = dataset['quality'].apply(lambda x: 'bueno' if x >= 7 else 'malo')
    X = dataset.drop(['quality', 'quality_label'], axis=1)
    y = dataset['quality_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#Red Neuronal Implementada
def MLP(X_train, X_test, y_train, y_test, op, hidden_layer_sizes=(100,50), max_iter=500):
    modelo = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    if op == 2:
        accuracy = accuracy_score(y_test, y_pred) 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=['malo', 'bueno']).ravel()
        specificity = tn / (tn + fp)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', pos_label='bueno')
    if op == 3:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_test, y_pred)

    print("\nMLP: ")
    print(f"Precisión: {precision}")
    print(f"Recuperación: {recall}")
    print(f"Especificidad: {specificity}")
    print(f"Puntuación F1: {f1}")

nombres_archivos = ['AutoInsurSweden.csv','wine-Quality.csv', 'pima-indians-diabetes.csv']

while True:
    print("\nDataSets:")
    for i, nombre_archivo in enumerate(nombres_archivos, start=1):
        print(f"{i}. {nombre_archivo}")

    opcion = input("Seleccione el dataSet: ('s' para terminar): ")

    if opcion.lower() == 's':
        break

    try:
        opcion = int(opcion)
        if opcion == 1:
            print("\nAutoInsurSweden.csv")
            X_train, X_test, y_train, y_test = AutoInsurSweden()
            regresion_logistica(X_train, X_test, y_train, y_test, opcion)
            k_vecinos_cercanos(X_train, X_test, y_train, y_test, opcion)
            maquinas_vector_soporte(X_train, X_test, y_train, y_test, opcion)
            naive_Bayes(X_train, X_test, y_train, y_test, opcion)
            MLP(X_train, X_test, y_train, y_test, opcion)
        elif opcion == 2:
            print("\nwine-Quality.csv")
            X_train, X_test, y_train, y_test = wine_Quality()
            regresion_logistica(X_train, X_test, y_train, y_test, opcion)
            k_vecinos_cercanos(X_train, X_test, y_train, y_test, opcion)
            maquinas_vector_soporte(X_train, X_test, y_train, y_test, opcion)
            naive_Bayes(X_train, X_test, y_train, y_test, opcion)
            MLP(X_train, X_test, y_train, y_test, opcion)
        elif opcion == 3:
            print("\npima-indians-diabetes.csv")
            X_train, X_test, y_train, y_test = pima_indians_diabetes()
            regresion_logistica(X_train, X_test, y_train, y_test, opcion)
            k_vecinos_cercanos(X_train, X_test, y_train, y_test, opcion)
            maquinas_vector_soporte(X_train, X_test, y_train, y_test, opcion)
            naive_Bayes(X_train, X_test, y_train, y_test, opcion)
            MLP(X_train, X_test, y_train, y_test, opcion)
        else:
            print("Opción inválida. Inténtelo de nuevo.")
    except ValueError:
        print("Ingrese una opción válida ('s' para salir)")