# Requisitos del Proyecto

Este proyecto requiere ciertas dependencias y datos para funcionar correctamente. A continuación se detalla todo lo necesario.

## Provisión de Datos (Importante)

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
