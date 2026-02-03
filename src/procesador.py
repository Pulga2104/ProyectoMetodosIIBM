import rasterio
from rasterio.merge import merge
import glob
import os

def verificar_o_crear_mapa(carpeta_datos, nombre_salida):
    """
    Verifica la existencia de un archivo de mapa unificado y lo genera si es necesario.
    
    Esta función comprueba si existe un archivo de mapa de elevación previamente procesado.
    Si el archivo no existe, busca archivos SRTM (.hgt) en la carpeta especificada y los
    combina en un único archivo GeoTIFF.
    
    Args:
        carpeta_datos (str): Ruta del directorio que contiene los archivos .hgt de entrada
        nombre_salida (str): Ruta y nombre del archivo GeoTIFF de salida
    
    Returns:
        bool: True si el archivo existe o se generó exitosamente, False en caso de error
    """
    # Verificar si el archivo de salida ya existe para evitar reprocesamiento
    if os.path.exists(nombre_salida):
        print(f"Mapa encontrado: '{nombre_salida}'. Saltando generación.")
        return True

    print(f"El archivo '{nombre_salida}' no existe. Iniciando generación...")
    
    # Buscar archivos con extensión .hgt
    patron = os.path.join(carpeta_datos, "*.hgt")
    archivos_hgt = glob.glob(patron)
    
    # Si no se encuentran archivos en minúsculas, intentar con mayúsculas
    if not archivos_hgt:
        patron_mayus = os.path.join(carpeta_datos, "*.HGT")
        archivos_hgt = glob.glob(patron_mayus)

    # Validar que se encontraron archivos de entrada
    if not archivos_hgt:
        print(f"ERROR CRÍTICO: No hay archivos .hgt en la carpeta '{carpeta_datos}'.")
        return False

    print(f"Se encontraron {len(archivos_hgt)} archivos. Uniendo (esto tomará un momento)...")

    try:
        # Abrir todos los archivos raster para procesamiento
        src_files_to_mosaic = []
        for fp in archivos_hgt:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # Ejecutar el proceso de fusión de rasters en un único mosaico
        # merge() combina múltiples archivos raster en uno solo, manejando superposiciones
        mosaico, out_trans = merge(src_files_to_mosaic)

        # Copiar metadatos del primer archivo como base
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Actualizar metadatos con las características del mosaico resultante
        out_meta.update({
            "driver": "GTiff",                    # Formato de salida GeoTIFF
            "height": mosaico.shape[1],           # Altura en píxeles del mosaico
            "width": mosaico.shape[2],            # Ancho en píxeles del mosaico
            "transform": out_trans,               # Transformación afín para georreferenciación
            "dtype": 'int16',                     # Tipo de dato int16, estándar para elevación SRTM
            "nodata": -32768                      # Valor que representa ausencia de datos
        })

        # Escribir el mosaico al archivo de salida con los metadatos actualizados
        with rasterio.open(nombre_salida, "w", **out_meta) as dest:
            dest.write(mosaico)

        print(f"ÉXITO: Archivo '{nombre_salida}' creado correctamente.")
        
        # Cerrar todos los archivos de entrada para liberar recursos
        for src in src_files_to_mosaic:
            src.close()
            
        return True

    except Exception as e:
        # Capturar y reportar cualquier error durante el proceso de fusión
        print(f"Error durante la unión de mapas: {e}")
        return False