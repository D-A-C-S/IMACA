# IMACA
Identificación de maderas del Cauca

### Entornos de ejecución
## [1] Notebooks[Recomendado]
Ejemplo: [Clasificación de 2 especies en google colab](https://colab.research.google.com/github/D-A-C-S/IMACA/blob/baseline/ejemplo(colab).ipynb)

## [2] Docker-CPU local
- Clonar repositorio
- Copiar imagenes a la carpeta datos
- Situarse en .../IMACA
- Ejecutar la imagen de docker:
```bash
sudo docker run -it --rm --mount type=bind,\
source="$(pwd)",target=/user/src/app \
lekodaca/effnet_cpu_training:0.01
```

## [3] Pip-CPU local
- Clonar repositorio
- Instalar requirements.txt
- Modificar el archivo de configuración config.yaml
- Ejecutar python train.py

Los pesos del modelo y la tabla de confusión son guardados en la carpeta outputs despues del entrenamiento.
