# masters_thesis
Wind Power Prediction using Video Vision Transformers for spatial and temporal information. 

TO DO LIST 

ViT
ViViT

COMPROBAR MAE DE TARGET VS TARGET DESPLAZADA hecho! 
PROBAR DE STANDARSCALE Despues de meter el canal de target 
Comparar errores con test de wilcoxson
Add normalization info to report 
CROSS VALIDATION 

AÑADIR A TEST LAS ULTIMAS 8 HORAS DEL AÑO ANTERIOR (en video)
PROBAR TODAS LAS COORDENADAS  hecho! 

- Get all datasets 
- Add information about past POWER values to the model hecho! 
    - Extra pixel 
    - Extra channel full of same valued pixels (desplazadas) hecho! 
    - Encoder-Decoder architecture
- Add physics informed information (add v^3 channels)

- Analyze outliers and/or points with high MAE or MSE 
- Visualize data (time series and spatial) hecho! 
- Visualize predictions (time series) hecho! 

Physics informed machine learning models

RESULTADOS ACTUALES 

### Dumb Baseline: MAE 0.0410 #### 

ViT 2 encoders 2 heads: MAE 0.0704
ViViT 2 encoders 2 heads: MAE 0.0706

Pasr label informed ViT 2 encoders 2 heads: NMAE: 0.0580
Past label informed ViViT 2 encoders 2 heads: MAE 0.0403  GREAT!!!

check dimension of predictions

SOLICITUD PARA PRESENTAR EN DICIEMBRE 