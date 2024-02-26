import numpy as np
import matplotlib.pyplot as plt

#Función del perceptrón
def perceptron(entradas, pesos, sesgo):
    suma_ponderada = np.dot(entradas, pesos) + sesgo
    #Función de activación
    if suma_ponderada >= 0:
        return 1
    else:
        return 0

#Leer el archivo csv
def leer_patrones(archivo):
    datos = np.genfromtxt(archivo, delimiter=',')
    entradas = datos[:, :-1]
    salidas = datos[:, -1]
    return entradas, salidas

#Entrenamiento del perceptrón
def entrenamiento(entradas, salidas, tasa_aprendizaje, max_epocas, criterio_convergencia):
    num_entradas = entradas.shape[1]
    num_patrones = entradas.shape[0]
    
    pesos = np.random.rand(num_entradas)
    sesgo = np.random.rand()
    epocas = 0
    convergencia = False

    while epocas < max_epocas and not convergencia:
        convergencia = True
        for i in range(num_patrones):
            entrada = entradas[i]
            salida_deseada = salidas[i]
            salida_recibida = np.dot(pesos, entrada) + sesgo
            error = salida_deseada - salida_recibida
            
            if abs(error) > criterio_convergencia:
                convergencia = False
                pesos += tasa_aprendizaje * error * entrada
                sesgo += tasa_aprendizaje * error
        epocas += 1
    return pesos, sesgo

#Testear el perceptrón ya entrenado
def prueba_per(entradas, pesos, sesgo):
    salida_recibida = np.dot(entradas, pesos) + sesgo
    return np.sign(salida_recibida)

#Calcular la precisión
def precision(salidas_reales, salidas_predichas):
    predicciones_correctas = np.sum(salidas_reales == salidas_predichas)
    total_predicciones = len(salidas_reales)
    precision = predicciones_correctas / total_predicciones
    return precision

def graficar(entradas, salidas, pesos, sesgo):
    plt.figure(figsize=(8, 6))
    #Graficar patrones
    plt.scatter(entradas[:, 0], entradas[:, 1], c=salidas, s=100, cmap=plt.cm.coolwarm)
    
    #Graficacion con recta de separacion
    x_min, x_max = entradas[:, 0].min() - 1, entradas[:, 0].max() + 1
    y_min, y_max = entradas[:, 1].min() - 1, entradas[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = prueba_per(np.c_[xx.ravel(), yy.ravel()], pesos, sesgo)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, colors='g', linestyles=['-'], linewidths=3 , levels=[0])
    plt.title('Patrones y Línea de Separación')
    plt.xlabel('Entrada X1')
    plt.ylabel('Entrada X2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    #Lectura de patrones de entrenamiento y prueba desde archivos CSV
    archivo_entrenamiento = 'OR_trn.csv'
    archivo_prueba = 'OR_tst.csv'
    
    #Patrones de entrenamiento
    entradas_entrenamiento, salidas_entrenamiento = leer_patrones(archivo_entrenamiento)
    
    #Patrones de prueba
    entradas_p, salidas_p = leer_patrones(archivo_prueba)

    #Parametros de entrenamiento
    max_epocas = 100
    tasa_aprendizaje = 0.1
    criterio_convergencia = 0.01 
    
    #Entrenamiento
    pesos_entrenados, sesgo_entrenado = entrenamiento(entradas_entrenamiento, salidas_entrenamiento, tasa_aprendizaje, 
    max_epocas, criterio_convergencia)
    print("Se finalizo el entrenamiento del perceptron exitosamente")

    #Datos de prueba para probar el perceptrón
    prediccion_sal = prueba_per(entradas_p, pesos_entrenados, sesgo_entrenado)
    
    precision = precision(salidas_p, prediccion_sal)
    print("Precisión con los datos:", precision)
    
    print("Salidas en la prueba:")
    print(salidas_p)
    print("Prediccion de salidas por el perceptrón:")
    print(prediccion_sal)
    graficar(entradas_entrenamiento, salidas_entrenamiento, pesos_entrenados, sesgo_entrenado)