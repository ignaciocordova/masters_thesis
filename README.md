# masters_thesis
Wind Power Prediction using Video Vision Transformers for spatial and temporal information. 

TO DO LIST 

COMPROBAR MAE DE TARGET VS TARGET DESPLAZADA hecho! 
PROBAR DE STANDARSCALE Despues de meter el canal de target 
Comparar errores con test de wilcoxson
Add normalization info to report 
CROSS VALIDATION 

AÑADIR A TEST LAS ULTIMAS 8 HORAS DEL AÑO ANTERIOR (en video)
PROBAR TODAS LAS COORDENADAS  hecho! 

- Get all datasets 
- Add information about past POWER values to the model
    - Extra pixel 
    - Extra channel full of same valued pixels (desplazadas) 
    - Encoder-Decoder architecture
- Add physics informed information (add v^3 channels)

- Analyze outliers and/or points with high MAE or MSE 
- Visualize data (time series and spatial)
- Visualize predictions (time series)


PREGUNTAS 
- Fecha de presentación? Si no estoy matriculado lo puedo presentar en Septiembre?? 
- Memoria + paper? O solo uno de los dos? 
- Doctorado industrial? 



RESULTADOS ACTUALES 

Dumb Baseline: MAE 0.041
ViViT 2 encoders 2 heads: MAE 0.0706
Past label informed ViViT 2 encoders 2 heads: MAE 0.0403  GREAT!!!