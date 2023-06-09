PROBAR DE STANDARSCALE Despues de meter el canal de target 
Comparar errores con test de wilcoxson
Add normalization info to report 
CROSS VALIDATION 
grid search para transofrmer depth & heads 

AÑADIR A TEST LAS ULTIMAS 8 HORAS DEL AÑO ANTERIOR (en video)
Comprobar dimensiones de las predicciones

- Get all datasets 
- Add physics informed information (add v^3 channels)

- Analyze outliers and/or points with high MAE or MSE 

Physics informed machine learning models

RESULTADOS ACTUALES 

### Dumb Baseline: MAE 0.0410 #### 

ViT 2 encoders 2 heads: MAE 0.0704
ViViT 2 encoders 2 heads: MAE 0.0706

Pasr label informed ViT 2 encoders 2 heads: NMAE: 0.0580
Past label informed ViViT 2 encoders 2 heads: MAE 0.0403  GREAT!!!


Hechos: 

Visualize data (time series and spatial) hecho! 
Visualize predictions (time series) hecho! 
COMPROBAR MAE DE TARGET VS TARGET DESPLAZADA hecho! 
ViT
ViViT
PROBAR TODAS LAS COORDENADAS  hecho! 
Add information about past POWER values to the model hecho! 
Extra channel full of same valued pixels (desplazadas) hecho! 