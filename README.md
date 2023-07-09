Objetivos: 

Crear un programa de Python que entre el ViT con 2016 y lo valide con 2017. 
Aplicar early stopping si el validaliton loss no disminuye al cabo de 10 epochs.  
Utilizar cross validation para el número de encoders y el número de heads. 



Ahora mismo lo que tengo es que estoy entrenando en TOOOODO el dataset 2016+2017 
pero voy mirando 2018 para ver cuando parar. Eso es trampa parece ser. 


Revisar el bucle del loop de evaluacion (test)


PROBAR modelos más complejos que el ViT para 4,6 u 8 horas hacia el futuro. 



PROBAR DE STANDARSCALE Despues de meter el canal de target 
Comparar errores con test de wilcoxson
Add normalization info to report 
CROSS VALIDATION 
grid search para transofrmer depth & heads 

AÑADIR A TEST LAS ULTIMAS NUM_FRAMES HORAS DEL AÑO ANTERIOR (en video) hecho! 
Comprobar dimensiones de las predicciones hecho! 

- Get all datasets 
- Add physics informed information (add v^3 channels)
- Analyze outliers and/or points with high MAE or MSE 

Physics informed machine learning models

RESULTADOS ACTUALES

Experimentos con validacion para 15 de Julio 
Tabla con resultados para cada modelo, número de parámetros...

### Dumb Baseline: MAE 0.0410 #### 

ViT 2 encoders 2 heads: MAE 0.0704
ViViT 2 encoders 2 heads: MAE 0.0706

Pasr label informed ViT 2 encoders 2 heads: NMAE: 0.0580
Past label informed ViViT 2 encoders 2 heads: MAE 0.0403  GREAT!!!


Hechos: 

Cambiar la estructura de los datasets creados:
    Imágenes
    Videos
    Imágenes con power anteriores 
    Videos con power anteriores (en cada frame)
Crear datasets de test con la última imagen de train
Creat datasets de video test con los útlimos NUM_FRAMES de train 
Entrenar de forma más agresiva (haciendo "trampas") --> conseguido 0.02 NMAE !!!! 
Guardar modelos en formato .pt para poder compararlos entre ellos. 
Visualize data (time series and spatial) hecho! 
Visualize predictions (time series) hecho! 
COMPROBAR MAE DE TARGET VS TARGET DESPLAZADA hecho! 
ViT
ViViT
PROBAR TODAS LAS COORDENADAS  hecho! 
Add information about past POWER values to the model hecho! 
Extra channel full of same valued pixels (desplazadas) hecho! 