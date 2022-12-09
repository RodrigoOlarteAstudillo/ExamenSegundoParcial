import cv2
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")

FILENAME = 'Jit1.JPG'
ITERACIONES = 24
CLUSTERS = 4
FILASCOLUMNAS = 13
SIGMA = 15
#ADVERTENCIA: Este código solo sirve para la imagen original de tamaño  2729 x 2833
#DE OTRA FORMA SE DEBE MOVER LOS RUIDOS DE MANERA MANUAL
RUIDO_1 = 500
RUIDO_2 = 10000

#200
#1000

#500
#10000

def escoger_puntos_semilla(cantidad_clusters:int,imagen_vector):
    #obtiene la posicion de los centroides y envia a los elegidos aleatoriamente mediante un arreglo 
    imagen_vector = np.array([[0,0,255],[255,0,0],[0,255,0],[0,0,0]])
    return imagen_vector#[np.random.randint(0,longitud_vector,cantidad_clusters)]

def escoger_clusters(puntos,imagen_vector):
    #se obtiene la distancia por medio de la suma del error cuadratico 
    #y se asignaran los puntos que pertenecen a cada cluster dependiendo de los centroides
    suma_cuadratica_media_error = ((imagen_vector - puntos[:, np.newaxis]) ** 2).sum(axis=2)
    minimo_suma_cuadratica_media_error = suma_cuadratica_media_error.argmin(axis=0)
    return minimo_suma_cuadratica_media_error

def elegir_nuevos_centroides(centroides,clusters,imagenEnVector):
    #el for tendra el rango de el numero de centroides, o en otra manera, la cantidad de clusters
    for i in range(CLUSTERS): 
        #guarda todos los elementos de un cluster determinado
        vec_sub = imagenEnVector[clusters==i] #verdaderos y falsos 
        #regresa el promedio de los valores del vec_sub y asigna los nuevos 
        centroides[i] = np.mean(vec_sub,axis=0)
    #se retornan los nuevos centroides
    return centroides

def kmedios(imagen,cantidadCluster):
    #obtenemos las dimensiones de la imagen (colores)
    dimensiones = 3 #cambiar a imagen.shape[2] si no se sabe si es a color o no
    #combertimos la imagen MxL en un vector de una sola dimension 
    #donde contiene los puntos de la siguiente forma [[255,255,255],[0,104,125],etc]
    #es decir, seguardan los puntos en una tripleta la cual contiene la informacion de los 3 colores
    imagenEnVector = imagen.reshape(-1,dimensiones)
    imagenEnVector = imagenEnVector.astype(int)
    #obtenemos K puntos previamente conocidos de nuestra imagen para facilitar la segmentacion del rojo
    #en donde estos seran los centroides originales
    #reciba la cantidad de clusters y mi vector de una dimension
    centroides = escoger_puntos_semilla(cantidadCluster,imagenEnVector)
    #creamos una nueva lista la cual contendra el cluster asignado dependiendo de los originales
    #es decir, si tenemos 3 centroides retornara un arreglo de longitud de los pixeles 
    #con valores de 0 a 2 [0,0,0,0,1,1,2,1,1,0,1,...]
    #y cada valor dependera de los 3 centroides elegidos anteriormente
    #el 0 es el punto en la primera posicion de la lista de centroides, el 1 el segundo y asi sucesivamente
    clusters = escoger_clusters(centroides,imagenEnVector)



    contador = 0
    while(ITERACIONES > contador):
        #print(contador)
        contador +=1
        #elegimos nuevos clusters 
        clusters = escoger_clusters(centroides,imagenEnVector)
        #se eligen nuevos centroides 
        centroides = elegir_nuevos_centroides(centroides,clusters,imagenEnVector)
        
    #el vector de una dimension lo volvelos a convertir en una imagen de NxMx3
    imagen_kmedios = centroides[clusters].reshape(imagen.shape)
    return imagen_kmedios.astype(np.uint8)
    # cv2.namedWindow("imagen original", cv2.WINDOW_NORMAL)
    # cv2.imshow("imagen original",imagen)
    # cv2.namedWindow("imagen k-medios", cv2.WINDOW_NORMAL)
    # cv2.imshow("imagen k-medios",imagen_kmedios.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()'

def prueba(imagen):
    #obtenemos las filas y columnas de la imagen binaria en azul
    filas,columnas = imagen.shape
    #debido a que es la matriz de escala de grises en azul, el rojo estara muy cerca del valor 0, por lo que 
    #mi umbral debe ser bajo pero no tanto, es decir, se optó por el valor de 40
    ret,thresh1 = cv2.threshold(imagen,40,255,cv2.THRESH_BINARY)

    cv2.imwrite("examen2_BlancoYNegro.jpg",thresh1)
    imagen2 = thresh1.copy()
    #para eliminar las manchas blancas que tienen los jitomates por el brillo, lo que se hace es primero hacer una copia 
    #de mi binaria, despues pintar unicamente el fondo exterior del mismo color que los tomates 
    cv2.floodFill(imagen2, None, (0, 0), (0, 0, 0))
    cv2.imwrite("examen2_BlancoYNegro_fondo.jpg",imagen2)
    #despues se invierten estos valores para que las manchas blancas ahora sean de color negro
    for i in range(filas):
        for j in range(columnas):
            if(imagen2[i,j]==255):
                imagen2[i,j]=0 
            else:
                imagen2[i,j]=255
            if(imagen2[i,j]==0): #y se ingresan los valore en la primera imagen para que ya no haya manchas blancas 
                thresh1[i,j]=0
    cv2.imwrite("examen2_manchas.jpg",imagen2)
    
    # print("voy aqui")
    # for i in range(filas):
    #     for j in range(columnas):
    #         print(i,j)
    #         if(thresh1[i,j]==255):
    #             cv2.floodFill(thresh1,None,(i,j),(random.uniform(50,190)))

            

    

    return thresh1

def tagging(imagen):
    #obtenemos la imagen ya segmentada en binario y le hacemos una copia
    imagen1 = imagen.copy()
    filas,columnas = imagen.shape
    gris = 50
    colores = list()
    contador = 0
    #el buscaddor de izquierda a derecha y de arriba a abajo, si detecta un color negro (valor 0) este cambiara 
    #a este pixel y a todos los pixeles negros que esten junto a el a un tono de gris X, una vez que se haga esto el gris aumentara
    #en 1 para que cada imagen tenga un color unico de pixeles, este color nos servira para las etiquetas
    for i in range(filas):
        for j in range(columnas):
            if(imagen1[i,j]==0):
                    contador +=1
                    #imagen1[i+4,j]=100
                    colores.append(gris)
                    cv2.floodFill(imagen1,None,(j,i),gris)
                    gris+=1
    # cv2.imshow("imagen1",imagen1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return imagen1,contador,colores
    #funcion que nos sirve para obtener el producto cruz de dos puntos o vectores
def calcularProductoCruz(punto1:list,punto2:list) -> list:
    productoCruz = [punto1[1]*punto2[2] - punto1[2]*punto2[1],-1*(punto1[0]*punto2[2] - punto1[2]*punto2[0]),punto1[0]*punto2[1] - punto1[1]*punto2[0]] 
    return productoCruz

def distancia(imagen,contador_colores,colores):
    filas,columnas = imagen.shape
    lista_clusters= list()
    lista_clusters_final = list()
    lista_clusters_completos = list()
    lista_clusters_completos_final = list()
    #creamos previamente las listas de listas en donde  almacenaremos los pixeles de cada tomate 
    for i in range(contador_colores):
        cluster = list()
        cluster2 = list()
        lista_clusters.append(cluster)
        lista_clusters_completos.append(cluster2)
    #aqui en una lista de listas guardaremos todos los pixeles que sirven de borde para los obtejos 
    #y en en la lista de clusters completos se guardaran todos los pixeles del tomate, independientemente 
    #si son bordes o no
    for i in range(filas):
        for j in range(columnas):
            adyacentes = False
            adyacentes = (imagen[i-1,j-1]==255 or imagen[i-1,j]==255 or imagen[i,j-1]==255 or imagen[i+1,j+1]==255 or imagen[i+1,j]==255 or imagen[i,j+1]==255 or imagen[i-1,j+1]==255 or imagen[i+1,j-1]==255)
            for index,x in enumerate(colores):
                if (imagen[i,j]==x) :
                    lista_clusters_completos[index].append((i,j))
                    if (adyacentes):
                        #print(imagen[i,j])
                        #print((imagen[i-1,j-1]==255 or imagen[i-1,j]==255 or imagen[i,j-1]==255 or imagen[i+1,j+1]==255 or imagen[i+1,j]==255 or imagen[i,j+1]==255 or imagen[i-1,j+1]==255 or imagen[i+1,j-1]==255))
                        #print(adyacentes)
                        #print(imagen[i-1,j-1]==255,imagen[i-1,j]==255,imagen[i,j-1]==255,imagen[i+1,j+1]==255,imagen[i+1,j]==255,imagen[i,j+1]==255,imagen[i-1,j+1]==255,imagen[i+1,j-1]==255,"->",i,j)
                        lista_clusters[index].append((i,j))
                        #imagen[i,j]=0
                    

    # cv2.imshow("hola",imagen)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()
            
    #lsita_cluster tiene todos mis pixeles de ese lugar 
    #lista[2]
    #print(len(lista_clusters))
    #en estos dos for se eliminan los clusters que tengan menos de una cantidad determinada de pixeles,
    #esto debido a que son ruido
    #por lo tanto este programa sirve para la imagen original, por lo que no funcionara
    #correctamente si se pone otra medida a menos que se cambien las constantes del ruido_1 y ruido_2 manualmente
    for i in (lista_clusters):
        #print(len(i))
        if(len(i)>RUIDO_1): #cambiar para el tamanio de la imagen
            lista_clusters_final.append(i)
    #print(len(lista_clusters_completos))
    for i in lista_clusters_completos:
        #print(len(i))
        if(len(i)>RUIDO_2):
            lista_clusters_completos_final.append(i)
    pixeles_distancias_maximas = list()
    distancias_maximas = 0
    print("primera imagen linea")
    #como solo nos interesa las lineas del segundo y del ultimo, entonces guardamos sus valores 
    #para procesarlos de manera mas facil
    segundo_cluster = lista_clusters_final[1]
    ultimo_cluster = lista_clusters_final[-1]
    #calculamos la distancia de cada pixel con los demas pixeles y se guarada la distancia mas grande junto con los 
    #pixeles que la obtuvieron del tomate 2 
    for i in range(0,len(segundo_cluster)):
        posiciones_1 = segundo_cluster[i]
        #print(posiciones_1[0],posiciones_1[1])
        for j in range(i,len(segundo_cluster)):
            posiciones_2 = segundo_cluster[j]
            #print(" ",posiciones_2[0],posiciones_2[1])
            distancia_actual = ecuacion_distancia(posiciones_1[0],posiciones_1[1],posiciones_2[0],posiciones_2[1])
            if(distancia_actual>distancias_maximas and posiciones_1[0]==posiciones_2[0]):
                distancias_maximas = distancia_actual
                pixeles_distancias_maximas = ((posiciones_1[0],posiciones_1[1]),(posiciones_2[0],posiciones_2[1]))
    print("Punto1:",pixeles_distancias_maximas[0]," punto2:",pixeles_distancias_maximas[1])
    print("La longitud de la primera recta es de:",distancias_maximas)
    pixeles_1 = pixeles_distancias_maximas[0]
    pixeles_2 = pixeles_distancias_maximas[1]
    pixeles_distancias_maximas = list()
    distancias_maximas = 0
    #lo mismo que la funcion anterior pero de ltomate 4
    print("segunda imagen linea")
    #print(distancias_maximas)
    for i in range(0,len(ultimo_cluster)):
        posiciones_1 = ultimo_cluster[i]
        #print(posiciones_1[0],posiciones_1[1])
        for j in range(i,len(ultimo_cluster)):
            posiciones_2 = ultimo_cluster[j]
            #print(" ",posiciones_2[0],posiciones_2[1])
            distancia_actual = ecuacion_distancia(posiciones_1[0],posiciones_1[1],posiciones_2[0],posiciones_2[1])
            if(distancia_actual>distancias_maximas):
                distancias_maximas = distancia_actual
                pixeles_distancias_maximas = ((posiciones_1[0],posiciones_1[1]),(posiciones_2[0],posiciones_2[1]))
    print("Punto1:",pixeles_distancias_maximas[0]," punto2:",pixeles_distancias_maximas[1])
    print("La longitud de la segunda recta es de:",distancias_maximas)
    pixeles_3 = pixeles_distancias_maximas[0]
    pixeles_4 = pixeles_distancias_maximas[1]
    #print(pixeles_distancias_maximas)
    
    line_thickness = 2
    #cv2.line(imagen, (pixeles_1[1],pixeles_1[0]), (pixeles_2[1],pixeles_2[0]), (100), thickness=2)

    #pasamos a una variable todos los pixeles de los tomates del segundo y ultimo, esto nos servira para obtener los valores del
    #producto cruz
    segundo_cluster = lista_clusters_completos_final[1]
    ultimo_cluster = lista_clusters_completos_final[-1]
    #obtenemos los valores de los puntos que forman la mayor recta de ambos tomates (1,2) y (3,4)
    punto1 = (pixeles_1[0],pixeles_1[1],1)
    punto2 = (pixeles_2[0],pixeles_2[1],1)
    punto3 = (pixeles_3[0],pixeles_3[1],1)
    punto4 = (pixeles_4[0],pixeles_4[1],1)
    #se obtiene el producto cruz de  ambos grupos dew puntos
    producto_cruz = calcularProductoCruz(punto1,punto2)
    producto_cruz_2 = calcularProductoCruz(punto3,punto4)
    print(producto_cruz_2)

    lista_lineas_1 = list()
    lista_lineas_2 = list()
    #si se cumple la ecuacion con los puntos de cada tomate, entonces significa que ese pixel forma parte de la recta y se guarda
    #su posicion
    #esto se hace para ambos tomates
    for i in segundo_cluster: #(192, 351)
        resultado = (i[0]*producto_cruz[0]) + (i[1]*producto_cruz[1]) + producto_cruz[2]
        if(resultado==0):
            lista_lineas_1.append(i)
            #imagen[i] = 255;
    resultado = 0
    for i in ultimo_cluster:
        resultado = (i[0]*producto_cruz_2[0]) + (i[1]*producto_cruz_2[1]) + producto_cruz_2[2]
        if(resultado==0):
            #print("hola")
            for j in range(-3,2):
                for k in range(-3,2):
                    lista_lineas_1.append((i[0] + j,i[1] + k))
                    imagen[i[0] +j,i[1] + k]=255
    #cv2.imwrite("255.jpg",imagen)


    return pixeles_1,pixeles_2,pixeles_3,pixeles_4,lista_lineas_1
#ecuacion que nos sirve para obtener la longitud de las lineas
def ecuacion_distancia(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def main():
    imagen = cv2.imread(FILENAME)
    filas,columnas,colores = imagen.shape
    print("tamanio de la imagen:",imagen.shape)
    #b,g,r = cv2.split(imagen)

    #kernelGaussiano =  kernelGauss(FILASCOLUMNAS,SIGMA)
    # print("bordes")
    # imagenBordes_b,imagenBordes_g,imagenBordes_r = obtenerImagenBordes(b,g,r,FILASCOLUMNAS)
    # imagenFiltroGauss_b = np.zeros((filas,columnas), dtype=np.uint8)
    # imagenFiltroGauss_g = np.zeros((filas,columnas), dtype=np.uint8)
    # imagenFiltroGauss_r = np.zeros((filas,columnas), dtype=np.uint8)
    # print("filtros")
    # filtro_b,filtro_g,filtro_r = filtro(kernelGaussiano,imagenBordes_b,imagenBordes_g,imagenBordes_r,FILASCOLUMNAS,imagenFiltroGauss_b,imagenFiltroGauss_g,imagenFiltroGauss_r,b)
    

    #image_merge = cv2.merge([filtro_b, filtro_g, filtro_r])
    #cv2.imwrite("examen2_blur.jpg",image_merge)
    print("kmedios")
    #obtenemos la imagen segmentada a color con ayuda de kmedios
    imagen_kmedios = kmedios(imagen,CLUSTERS)
    #guardamos esa imagen
    cv2.imwrite("examen2_kmedios.jpg",imagen_kmedios)
    #obtenemos unicamente la matriz de pixeles de azul 
    b,g,r = cv2.split(imagen_kmedios)
    print("mejorar blanco y negro")
    #la matriz azul esta a escala de grises, por lo tanto esa la llamaremos a la funcion llamada "prueba"
    #esta funcion lo que hace es obtener el binario de esa imagen, se envia la azul debido a que los pixeles en rojo
    #son los mas oscuros de la imagen, por lo que es mas sencillo obtener los tomates segmentados
    black_white = prueba(b)
    
    #se guarda la imagen segmentada en blanco y negro
    cv2.imwrite("examen2_BN.jpg",black_white)

    print("tag")
    #se hace una nueva imagen que etiquete las formas para despues agrupar y saber 
    #a que pixel equivale cada tomate 
    imagen_tags,contador_colores,colores = tagging(black_white)
    print("distancia")
    #la funcion "distancia" tiene varias funcionalidades, la primera es guardar en una lista 
    #los pixeles que sirven como bordes de cada tomate, recordando que estos siguen estando etiquetados.
    
    #Otra funcion es eliminar el ruido o los objetos que no sean tomates
    #por ultimo, se obtiene las distancias entre los bordes de cada tomate para obtener los extremos y se obtiene el 
    #producto cruz de los pixeles elegidos para guardar los valores de las lineas
    punto_1,punto_2,punto_3,punto_4,lineas_1 = distancia(imagen_tags,contador_colores,colores)
    #los puntos verdes en las lineas son de los valores obtenidos por el producto cruz
    #sin embargo, recordemos que aqui los puntos no pueden ser flotantes, por lo que en el ultimo tomate 
    #salen pocos verdes, que son los valores enteros que cumplieron con el producto cruz
    #Y por eso se pone la linea amarilla con la funcion de abajo, para enseñar que el producto cruz
    #si sigue esa linea
    cv2.line(imagen, (punto_1[1],punto_1[0]), (punto_2[1],punto_2[0]), (0,255,255), thickness=2)
    cv2.line(imagen, (punto_3[1],punto_3[0]), (punto_4[1],punto_4[0]), (0,255,255), thickness=2)
    for i in lineas_1:
        imagen[i] = (0,255,0)
    cv2.imwrite("examen2_final.jpg",imagen)

if __name__ == "__main__":
    main()