# Requisitos del Proyecto

Este proyecto requiere ciertas dependencias y datos para funcionar correctamente. A continuación se detalla todo lo necesario.

## Limitacion de provincias (Datos):

El archivo GeoJSON lo obtuvimos del proyecto GADM, la descarga de estos datos es unicamente para fines academicos y no comerciales, donde descargamos unicamente el nivel 1, ya que este solo contiene los datos de las limitaciones por provincias.

Link de GADM: https://gadm.org/download_country.html

## Provisión de Datos (Importante)

**Uso de datos del repositorio**

Para comprimir los archivos de gran tamaño (.hgt , .slt) se utilizo Git LFS.

Para ello primero debe de instalar LFS con la linea de comando: **git lfs install**

Despuesa de instalar LFS, al clonar el repositorio los archivos originales se descargaran del Github, 

Si es que ya tiene el repositorio de manera local y conectado remotamente al repositorio, debe de utlizar el comando: **git lfs pull**, para descargar los archivos comprimidos del repositorio.

**El sistema necesita datos de elevación del terreno para generar los modelos.**

1.  Descarga los datos SRTM de tu área de interés (archivos `.hgt`).
2.  Coloca los archivos descargados en la carpeta:
    `datos_srtm/`
3.  Alternativamente, si ya tienes un mapa procesado, el sistema buscará `ecuador_completo.tif`.

---

## Dependencias de Python

Las siguientes bibliotecas son necesarias para ejecutar el código.

### Detalle de las librerías

| Librería | Propósito en el proyecto |
| :--- | :--- |
| **`matplotlib`** | Motor principal de la interfaz gráfica. Maneja la visualización interactiva, controles, botones y la selección de áreas. |
| **`rasterio`** | Esencial para leer archivos geoespaciales (`.hgt`, `.tif`). Gestiona las coordenadas geográficas y la fusión de mapas. |
| **`numpy`** | Realiza todos los cálculos numéricos y matriciales de alto rendimiento sobre los datos de elevación. |
| **`numpy-stl`** | Se encarga de la generación de la malla 3D y el guardado final de los archivos `.stl` para impresión 3D. |
| **`scipy`** | Proporciona los algoritmos de filtrado (promedio y mediana) para suavizar las superficies rugosas. |
| **`contextily`** | Descarga y renderiza mapas de calles (OpenStreetMap) en el fondo para que sepas dónde estás seleccionando. |

## Ejecución

Una vez instaladas las dependencias y colocados los datos:

```bash
python Main.py
```

