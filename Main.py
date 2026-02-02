import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, RadioButtons, Button, TextBox
import rasterio
from rasterio.mask import mask as rasterio_mask
import numpy as np
from stl import mesh
from scipy.ndimage import uniform_filter, median_filter
import os
import datetime
import sys
import json
try:
    import geopandas as gpd
    from shapely.geometry import mapping
    TIENE_GEOPANDAS = True
except ImportError:
    TIENE_GEOPANDAS = False
    print("AVISO: Instala 'geopandas' para usar polígonos reales de provincias.")
    print("  pip install geopandas")

# Importación del módulo personalizado de procesamiento
from src.procesador import verificar_o_crear_mapa

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
CARPETA_DATOS = "datos_srtm"      
ARCHIVO_FINAL = "ecuador_completo.tif"
CARPETA_STL = "modelos_stl"
ARCHIVO_PROVINCIAS = "gadm41_ECU_1.json"  # GeoJSON con polígonos de provincias

# Verificación de la existencia o generación del mapa de elevación
print("--- INICIANDO SISTEMA ---")
mapa_listo = verificar_o_crear_mapa(CARPETA_DATOS, ARCHIVO_FINAL)

if not mapa_listo:
    print("No podemos continuar sin el mapa. Revisa la carpeta de datos.")
    sys.exit() 

# Intento de importación de Contextily para visualización de mapas base
try:
    import contextily as cx
    TIENE_MAPA_REAL = True
except ImportError:
    TIENE_MAPA_REAL = False
    print("AVISO: Instala 'contextily' para ver el mapa de calles.")

# ==========================================
# ALGORITMOS DE SUAVIZADO
# ==========================================

def suavizado_promedio(matriz):
    """
    Aplica filtro de promedio (Mean Filter) para suavizado de datos.
    
    Este algoritmo reduce el ruido calculando el promedio de los píxeles vecinos
    en una ventana de 3x3, resultando en una superficie más suave.
    
    Args:
        matriz (numpy.ndarray): Matriz de elevación a suavizar
    
    Returns:
        numpy.ndarray: Matriz suavizada mediante promedio
    """
    return uniform_filter(matriz, size=3)

def suavizado_mediana(matriz):
    """
    Aplica filtro de mediana (Median Filter) para eliminación de ruido.
    
    Este algoritmo reemplaza cada píxel con la mediana de sus vecinos en una
    ventana de 3x3, preservando mejor los bordes que el filtro de promedio.
    
    Args:
        matriz (numpy.ndarray): Matriz de elevación a filtrar
    
    Returns:
        numpy.ndarray: Matriz filtrada mediante mediana
    """
    return median_filter(matriz, size=3)

# ==========================================
# GESTIÓN DE PROVINCIAS
# ==========================================

def cargar_provincias():
    """
    Carga el archivo GeoJSON con los polígonos de provincias de Ecuador.
    
    Returns:
        GeoDataFrame: GeoDataFrame con provincias y sus geometrías (polígonos)
    """
    if not os.path.exists(ARCHIVO_PROVINCIAS):
        print(f"AVISO: Archivo '{ARCHIVO_PROVINCIAS}' no encontrado.")
        print("Descarga GADM level1 para Ecuador desde: https://gadm.org/download_country.html")
        return None
    
    if not TIENE_GEOPANDAS:
        print("ERROR: Se requiere geopandas para cargar provincias.")
        print("Instala con: pip install geopandas")
        return None
    
    try:
        # Cargar GeoJSON con geopandas
        gdf = gpd.read_file(ARCHIVO_PROVINCIAS)
        
        # El campo de nombre de provincia puede ser 'NAME_1' o similar en GADM
        if 'NAME_1' in gdf.columns:
            gdf = gdf.rename(columns={'NAME_1': 'provincia'})
        elif 'NOMBRE' in gdf.columns:
            gdf = gdf.rename(columns={'NOMBRE': 'provincia'})
        
        # Excluir Galápagos (no está en el mapa de elevación)
        gdf = gdf[~gdf['provincia'].str.contains('Galápagos|Galapagos', case=False, na=False)]
        
        print(f"✓ Provincias cargadas: {len(gdf)} encontradas (sin Galápagos).")
        print(f"  Provincias: {', '.join(sorted(gdf['provincia'].unique()))}")
        return gdf
    
    except Exception as e:
        print(f"Error cargando provincias: {e}")
        return None

def generar_stl_provincia(provincia_nombre, provincias_gdf, metodo='Promedio', guardar_preview=True):
    """
    Genera un modelo STL para una provincia específica de Ecuador.
    
    Args:
        provincia_nombre (str): Nombre de la provincia
        provincias_gdf (GeoDataFrame): GeoDataFrame con geometrías de provincias
        metodo (str): Método de suavizado ('Promedio' o 'Mediana')
        guardar_preview (bool): Si True, guarda una imagen de verificación del recorte
    """
    # Buscar la provincia (case-insensitive)
    prov_row = provincias_gdf[provincias_gdf['provincia'].str.lower() == provincia_nombre.lower()]
    
    if prov_row.empty:
        print(f"Provincia '{provincia_nombre}' no encontrada.")
        return
    
    ahora = datetime.datetime.now()
    timestamp = ahora.strftime("%H-%M-%S")
    print(f"\n[{timestamp}] Procesando provincia: {provincia_nombre}")
    
    try:
        with rasterio.open(ARCHIVO_FINAL) as src:
            # Asegurar que la geometría esté en el mismo CRS que el raster
            if prov_row.crs != src.crs:
                prov_row_reproj = prov_row.to_crs(src.crs)
                geom = prov_row_reproj.geometry.values[0]
            else:
                geom = prov_row.geometry.values[0]
            
            # Recortar raster usando el polígono de la provincia
            out_image, out_transform = rasterio_mask(src, [mapping(geom)], crop=True, nodata=-32768)
            matriz = out_image[0]
            
            # Validación de tamaño mínimo de selección
            if matriz.size < 100:
                print(f"  ⚠ Área muy pequeña para {provincia_nombre}.")
                return
            
            # VERIFICACIÓN VISUAL: Guardar imagen del recorte con contorno
            if guardar_preview:
                carpeta_preview = "verificacion_provincias"
                if not os.path.exists(carpeta_preview):
                    os.makedirs(carpeta_preview)
                
                fig_verify, ax_verify = plt.subplots(figsize=(10, 8))
                
                # Mostrar matriz de elevación
                data_preview = matriz.copy().astype(float)
                data_preview[data_preview < -1000] = np.nan
                
                im = ax_verify.imshow(data_preview, cmap='terrain', interpolation='bilinear')
                plt.colorbar(im, ax=ax_verify, label='Elevación (m)')
                
                # Dibujar contorno de la provincia sobre el recorte
                from rasterio.plot import plotting_extent
                extent_crop = plotting_extent(out_image[0], out_transform)
                
                # Convertir geometría a coordenadas de imagen
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    # Transformar coordenadas geográficas a índices de píxeles
                    from affine import Affine
                    inv_transform = ~out_transform
                    xs, ys = [], []
                    for xi, yi in zip(x, y):
                        col, row = inv_transform * (xi, yi)
                        xs.append(col)
                        ys.append(row)
                    ax_verify.plot(xs, ys, 'r-', linewidth=2, label='Límite provincial')
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        x, y = poly.exterior.xy
                        inv_transform = ~out_transform
                        xs, ys = [], []
                        for xi, yi in zip(x, y):
                            col, row = inv_transform * (xi, yi)
                            xs.append(col)
                            ys.append(row)
                        ax_verify.plot(xs, ys, 'r-', linewidth=2)
                
                ax_verify.set_title(f'VERIFICACIÓN: {provincia_nombre}\n'
                                   f'Área: {matriz.shape[0]}x{matriz.shape[1]} píxeles\n'
                                   f'Elevación: {np.nanmin(data_preview):.0f}m - {np.nanmax(data_preview):.0f}m',
                                   fontsize=12, fontweight='bold')
                ax_verify.set_xlabel('Columnas (píxeles)')
                ax_verify.set_ylabel('Filas (píxeles)')
                ax_verify.legend()
                ax_verify.grid(True, alpha=0.3)
                
                # Guardar imagen de verificación
                nombre_limpio = provincia_nombre.replace(' ', '_').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
                preview_filename = os.path.join(carpeta_preview, f"Verificacion_{nombre_limpio}_{timestamp}.png")
                plt.savefig(preview_filename, dpi=150, bbox_inches='tight')
                plt.close(fig_verify)
                print(f"  ✓ Verificación guardada: {preview_filename}")
            
            # Limpieza de datos
            matriz = matriz.astype(float)
            matriz[matriz < -1000] = np.nan
            matriz = np.nan_to_num(matriz, nan=np.nanmin(matriz))
            
            # Aplicar suavizado
            if 'Promedio' in metodo:
                matriz_suave = suavizado_promedio(matriz)
                matriz_suave = suavizado_promedio(matriz_suave)
                matriz_suave = suavizado_promedio(matriz_suave)
                etiqueta = "Promedio"
            else:
                matriz_suave = suavizado_mediana(matriz)
                matriz_suave = suavizado_mediana(matriz_suave)
                etiqueta = "Mediana"
            
            # Generar nombre de archivo sin caracteres especiales
            nombre_limpio = provincia_nombre.replace(' ', '_').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            nombre_archivo = f"Modelo_{nombre_limpio}_{etiqueta}_{timestamp}.stl"
            
            # Generar STL
            generar_stl(matriz_suave, nombre_archivo, metodo)
            print(f"  ✓ Modelo de {provincia_nombre} generado exitosamente.")
            
    except Exception as e:
        print(f"  ✗ Error generando modelo para {provincia_nombre}: {e}")

def generar_todas_provincias(provincias_gdf, metodo='Promedio'):
    """
    Genera modelos STL para todas las provincias de Ecuador.
    
    Args:
        provincias_gdf (GeoDataFrame): GeoDataFrame con información de provincias
        metodo (str): Método de suavizado a aplicar
    """
    if provincias_gdf is None:
        print("No hay datos de provincias cargados.")
        return
    
    print(f"\n{'='*70}")
    print(f"GENERANDO MODELOS PARA TODAS LAS PROVINCIAS DE ECUADOR")
    print(f"Método de suavizado: {metodo}")
    print(f"Total de provincias: {len(provincias_gdf)}")
    print(f"{'='*70}\n")
    
    total = len(provincias_gdf)
    for idx, (_, row) in enumerate(provincias_gdf.iterrows(), 1):
        provincia = row['provincia']
        print(f"[{idx}/{total}] Procesando: {provincia}...")
        generar_stl_provincia(provincia, provincias_gdf, metodo)
    
    print(f"\n{'='*70}")
    print("✓ PROCESO COMPLETADO - Todos los modelos generados")
    print(f"{'='*70}\n")

def dibujar_limites_provincias(ax, provincias_gdf):
    """
    Dibuja los límites reales de todas las provincias en el mapa.
    
    Args:
        ax: Eje de matplotlib donde dibujar
        provincias_gdf (GeoDataFrame): GeoDataFrame con geometrías de provincias
    """
    if provincias_gdf is None:
        return
    
    # Reproyectar si es necesario (al CRS del raster que está en EPSG:4326)
    try:
        with rasterio.open(ARCHIVO_FINAL) as src:
            if provincias_gdf.crs != src.crs:
                provincias_gdf = provincias_gdf.to_crs(src.crs)
    except Exception as e:
        print(f"Advertencia al verificar CRS: {e}")
    
    for _, row in provincias_gdf.iterrows():
        nombre = row['provincia']
        geom = row['geometry']
        
        # Dibujar el polígono real de la provincia
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.plot(x, y, color='#FF5722', linewidth=1.5, alpha=0.7, linestyle='-')
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='#FF5722', linewidth=1.5, alpha=0.7, linestyle='-')
        
        # Agregar nombre de provincia en el centroide
        centroid = geom.centroid
        ax.text(centroid.x, centroid.y, nombre, 
                fontsize=7, ha='center', va='center',
                color='#D32F2F', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#FF5722', linewidth=1))

# ==========================================
# GENERACIÓN DE MODELOS STL
# ==========================================

def generar_stl(matriz, nombre_archivo, metodo_usado):
    """
    Genera un modelo STL sólido cerrado a partir de datos de elevación.
    
    Esta función crea un modelo 3D completo con superficie superior (topografía),
    base inferior plana y paredes laterales, guardándolo en la carpeta designada.
    
    Args:
        matriz (numpy.ndarray): Matriz de datos de elevación
        nombre_archivo (str): Nombre del archivo STL de salida
        metodo_usado (str): Método de suavizado aplicado (para referencia)
    """
    
    # Crear el directorio de salida si no existe
    if not os.path.exists(CARPETA_STL):
        os.makedirs(CARPETA_STL)
        print(f"Carpeta '{CARPETA_STL}' creada.")
    
    # Construir ruta completa del archivo de salida
    ruta_completa = os.path.join(CARPETA_STL, nombre_archivo)

    # Parámetros de escalado y dimensiones del modelo
    tamaño_base_mm = 120           # Tamaño base del modelo en milímetros
    exageracion_vertical = 0.04    # Factor de exageración vertical para relieve
    altura_base_mm = 5.0           # Altura de la base sólida en milímetros
    
    filas, columnas = matriz.shape
    x = np.arange(0, columnas)
    y = np.arange(0, filas)
    X, Y = np.meshgrid(x, y)
    
    # Normalización de elevaciones y cálculo de escala
    Z_norm = matriz - np.nanmin(matriz)
    scale = tamaño_base_mm / max(filas, columnas)
    
    # Aplicar escalado a coordenadas
    X = X * scale
    Y = Y * scale
    Z_top = (Z_norm * scale * exageracion_vertical) + altura_base_mm
    Z_bottom = np.zeros_like(Z_top)

    # Cálculo del número total de triángulos necesarios
    n_tris_top = (filas - 1) * (columnas - 1) * 2
    n_tris_walls = (2 * (filas - 1) + 2 * (columnas - 1)) * 2
    total_caras = n_tris_top * 2 + n_tris_walls
    datos = np.zeros(total_caras, dtype=mesh.Mesh.dtype)
    idx = 0 

    # A) Construcción de la superficie superior (topografía)
    v0_x = X[:-1, :-1].flatten()
    v0_y = Y[:-1, :-1].flatten()
    v0_z = Z_top[:-1, :-1].flatten()
    v1_x = X[:-1, 1:].flatten()
    v1_y = Y[:-1, 1:].flatten()
    v1_z = Z_top[:-1, 1:].flatten()
    v2_x = X[1:, :-1].flatten()
    v2_y = Y[1:, :-1].flatten()
    v2_z = Z_top[1:, :-1].flatten()
    v3_x = X[1:, 1:].flatten()
    v3_y = Y[1:, 1:].flatten()
    v3_z = Z_top[1:, 1:].flatten()
    
    # Asignación de triángulos para la superficie superior
    end = idx + n_tris_top
    datos['vectors'][idx:end:2, 0] = np.column_stack((v0_x, v0_y, v0_z))
    datos['vectors'][idx:end:2, 1] = np.column_stack((v1_x, v1_y, v1_z))
    datos['vectors'][idx:end:2, 2] = np.column_stack((v2_x, v2_y, v2_z))
    datos['vectors'][idx+1:end:2, 0] = np.column_stack((v3_x, v3_y, v3_z))
    datos['vectors'][idx+1:end:2, 1] = np.column_stack((v2_x, v2_y, v2_z))
    datos['vectors'][idx+1:end:2, 2] = np.column_stack((v1_x, v1_y, v1_z))
    idx = end

    # B) Construcción de la superficie inferior (base plana)
    v0_z_b = Z_bottom[:-1, :-1].flatten()
    v1_z_b = Z_bottom[:-1, 1:].flatten()
    v2_z_b = Z_bottom[1:, :-1].flatten()
    v3_z_b = Z_bottom[1:, 1:].flatten()
    
    # Asignación de triángulos para la base
    end = idx + n_tris_top
    datos['vectors'][idx:end:2, 0] = np.column_stack((v0_x, v0_y, v0_z_b))
    datos['vectors'][idx:end:2, 1] = np.column_stack((v2_x, v2_y, v2_z_b)) 
    datos['vectors'][idx:end:2, 2] = np.column_stack((v1_x, v1_y, v1_z_b)) 
    datos['vectors'][idx+1:end:2, 0] = np.column_stack((v3_x, v3_y, v3_z_b))
    datos['vectors'][idx+1:end:2, 1] = np.column_stack((v1_x, v1_y, v1_z_b)) 
    datos['vectors'][idx+1:end:2, 2] = np.column_stack((v2_x, v2_y, v2_z_b)) 
    idx = end

    # C) Construcción de paredes laterales
    def agregar_pared(x_edge, y_edge, z_top_edge, z_bot_edge):
        """
        Genera triángulos para una pared lateral del modelo.
        
        Args:
            x_edge (numpy.ndarray): Coordenadas X del borde
            y_edge (numpy.ndarray): Coordenadas Y del borde
            z_top_edge (numpy.ndarray): Coordenadas Z del borde superior
            z_bot_edge (numpy.ndarray): Coordenadas Z del borde inferior
        """
        nonlocal idx
        n = len(x_edge) - 1
        for i in range(n):
            p1 = [x_edge[i],   y_edge[i],   z_top_edge[i]]   
            p2 = [x_edge[i+1], y_edge[i+1], z_top_edge[i+1]] 
            p3 = [x_edge[i+1], y_edge[i+1], z_bot_edge[i+1]] 
            p4 = [x_edge[i],   y_edge[i],   z_bot_edge[i]]   
            datos['vectors'][idx] = [p1, p3, p4]
            idx += 1
            datos['vectors'][idx] = [p1, p2, p3]
            idx += 1

    # Generación de las cuatro paredes laterales
    agregar_pared(X[0, :], Y[0, :], Z_top[0, :], Z_bottom[0, :])
    agregar_pared(X[-1, ::-1], Y[-1, ::-1], Z_top[-1, ::-1], Z_bottom[-1, ::-1])
    agregar_pared(X[::-1, 0], Y[::-1, 0], Z_top[::-1, 0], Z_bottom[::-1, 0])
    agregar_pared(X[:, -1], Y[:, -1], Z_top[:, -1], Z_bottom[:, -1])

    # Escritura del archivo STL en disco
    malla = mesh.Mesh(datos)
    malla.save(ruta_completa)
    print(f"GUARDADO: {ruta_completa}")

# ==========================================
# INTERFAZ GRÁFICA INTERACTIVA
# ==========================================
radio = None
src_global = None
ax_global = None
basemap_img = None

def actualizar_mapa_base():
    """
    Actualiza el mapa base de calles según el nivel de zoom actual.
    
    Esta función recarga el mapa base de Contextily con mayor detalle
    cuando el usuario hace zoom, proporcionando una visualización nítida
    en cualquier nivel de acercamiento.
    """
    global basemap_img, ax_global, src_global
    
    if not TIENE_MAPA_REAL or ax_global is None or src_global is None:
        return
    
    try:
        # Obtener límites actuales de la vista
        xlim = ax_global.get_xlim()
        ylim = ax_global.get_ylim()
        
        # Calcular nivel de zoom basado en el área visible
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        area = width * height
        
        # Determinar zoom apropiado (más zoom = más detalle)
        if area < 0.1:
            zoom = 13
        elif area < 1:
            zoom = 11
        elif area < 5:
            zoom = 9
        else:
            zoom = 8
        
        # Remover mapa base anterior si existe
        if basemap_img is not None:
            basemap_img.remove()
        
        # Agregar nuevo mapa base con el nivel de zoom calculado
        basemap_img = cx.add_basemap(
            ax_global, 
            crs=src_global.crs.to_string(), 
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom=zoom,
            attribution=False
        )
        
        ax_global.figure.canvas.draw_idle()
        
    except Exception as e:
        print(f"Error actualizando mapa base: {e}")

def on_scroll(event):
    """
    Maneja el evento de desplazamiento de la rueda del ratón para zoom.
    
    Implementa zoom centrado en la posición del cursor y actualiza el mapa
    base para mantener la nitidez visual.
    
    Args:
        event: Evento de scroll de matplotlib
    """
    ax = event.inaxes
    if ax is None:
        return
    
    base_scale = 1.3
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    
    # Determinar dirección del zoom
    if event.button == 'up':
        scale_factor = 1 / base_scale
    elif event.button == 'down':
        scale_factor = base_scale
    else:
        scale_factor = 1
    
    # Calcular nuevos límites manteniendo la posición del cursor
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
    
    ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
    ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
    
    # Actualizar mapa base con nuevo nivel de detalle
    actualizar_mapa_base()

def al_seleccionar(eclick, erelease):
    """
    Procesa la selección rectangular del usuario y genera el modelo STL.
    
    Args:
        eclick: Evento de clic inicial (esquina del rectángulo)
        erelease: Evento de liberación del clic (esquina opuesta)
    """
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    lon_min, lon_max = sorted([x1, x2])
    lat_min, lat_max = sorted([y1, y2])
    
    # Generación de timestamp para nombre único de archivo
    ahora = datetime.datetime.now()
    timestamp = ahora.strftime("%H-%M-%S")
    print(f"\n[{timestamp}] Procesando selección...")
    opcion = radio.value_selected

    try:
        with rasterio.open(ARCHIVO_FINAL) as src:
            # Conversión de coordenadas geográficas a índices de píxel
            f_arr, c_izq = src.index(lon_min, lat_max)
            f_aba, c_der = src.index(lon_max, lat_min)
            
            # Limitación de índices a dimensiones válidas del raster
            f_arr = max(0, f_arr)
            c_izq = max(0, c_izq)
            f_aba = min(src.height, f_aba)
            c_der = min(src.width, c_der)
            
            # Definición de ventana de lectura y extracción de datos
            ventana = rasterio.windows.Window.from_slices((f_arr, f_aba), (c_izq, c_der))
            matriz = src.read(1, window=ventana)
            
            # Validación de tamaño mínimo de selección
            if matriz.size < 100:
                print("Selección muy pequeña.")
                return

            # Limpieza de datos: conversión de tipo y manejo de valores inválidos
            matriz = matriz.astype(float)
            matriz[matriz < -1000] = np.nan
            matriz = np.nan_to_num(matriz, nan=np.nanmin(matriz))

            # Aplicación del algoritmo de suavizado seleccionado
            if 'Promedio' in opcion:
                matriz_suave = suavizado_promedio(matriz) 
                matriz_suave = suavizado_promedio(matriz_suave)
                matriz_suave = suavizado_promedio(matriz_suave)
                etiqueta = "Promedio"
            else:
                matriz_suave = suavizado_mediana(matriz)
                matriz_suave = suavizado_mediana(matriz_suave)
                etiqueta = "Mediana"

            # Construcción del nombre de archivo con metadatos
            nombre_archivo = f"Modelo_{etiqueta}_Lat{abs(int(lat_min))}_{timestamp}.stl"
            generar_stl(matriz_suave, nombre_archivo, opcion)
            
            # Actualización de interfaz con confirmación
            ax_global.set_title(
                f"MAPA INTERACTIVO 3D - Generador de Modelos Topográficos\n"
                f"Archivo guardado: {nombre_archivo}", 
                fontsize=13, 
                fontweight='bold',
                color='#2E7D32',
                pad=15
            )
            plt.draw()

    except Exception as e:
        print(f"Error: {e}")

# ==========================================
# INICIALIZACIÓN Y LANZAMIENTO DE LA INTERFAZ
# ==========================================
print("Cargando interfaz...")

# Cargar provincias
provincias_dict = cargar_provincias()

try:
    with rasterio.open(ARCHIVO_FINAL) as src:
        src_global = src
        # Obtención de límites geográficos para configuración de ejes
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Reducción de resolución para vista previa inicial
        factor = 20
        h_new = int(src.height / factor)
        w_new = int(src.width / factor)
        data = src.read(1, out_shape=(h_new, w_new))

    # Configuración de la figura con estilo mejorado
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10), facecolor='#F5F5F5')
    
    # Ajuste de márgenes para mejor distribución (más espacio para controles)
    ax = plt.axes([0.05, 0.1, 0.60, 0.85])
    ax_global = ax
    
    # Aplicar estilo al área de trazado
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(1.5)

    # Visualización con mapa base de alta calidad
    if TIENE_MAPA_REAL:
        # Preparación de datos de elevación con transparencia
        data = data.astype(float)
        data[data < -1000] = np.nan
        
        # Capa de elevación semitransparente sobre mapa base
        ax.imshow(data, cmap='terrain', extent=extent, alpha=0.4, interpolation='bilinear')
        
        # Agregar mapa base inicial
        try:
            basemap_img = cx.add_basemap(
                ax, 
                crs=src_global.crs.to_string(), 
                source=cx.providers.OpenStreetMap.Mapnik,
                zoom=8,
                attribution=False
            )
        except Exception as e:
            print(f"Advertencia: No se pudo cargar el mapa base inicial: {e}")
    else:
        # Visualización simple de elevación sin mapa base
        data = data.astype(float)
        data[data < -1000] = np.nan
        ax.imshow(data, cmap='terrain', extent=extent, aspect='auto', interpolation='bilinear')

    # Dibujar límites de provincias - DESACTIVADO
    # if provincias_dict is not None:
    #     dibujar_limites_provincias(ax, provincias_dict)

    # Configuración de título y etiquetas con estilo profesional
    ax.set_title(
        "MAPA INTERACTIVO 3D - Generador de Modelos Topográficos de Ecuador\n"
        "MODO 1: Dibuja un rectángulo | MODO 2: Genera por provincia (ver panel derecho)", 
        fontsize=13, 
        fontweight='bold',
        color='#1976D2',
        pad=15
    )
    ax.set_xlabel("Longitud", fontsize=11, fontweight='bold', color='#424242')
    ax.set_ylabel("Latitud", fontsize=11, fontweight='bold', color='#424242')
    
    # Mejorar apariencia de los ticks
    ax.tick_params(colors='#424242', labelsize=10)

    # ===== PANEL DE CONTROLES =====
    
    # 1. Algoritmo de suavizado
    ejes_botones = plt.axes([0.67, 0.70, 0.30, 0.20], facecolor='#FAFAFA')
    ejes_botones.set_title(
        'Algoritmo de Suavizado', 
        fontsize=10, 
        fontweight='bold', 
        color='#1976D2',
        pad=10
    )
    
    # Configuración de botones de radio con mejor presentación
    radio = RadioButtons(
        ejes_botones, 
        ('Promedio\n(Más suave)', 'Mediana\n(Preserva bordes)'),
        activecolor='#1976D2'
    )
    
    # Estilizar los labels de los radio buttons
    for label in radio.labels:
        label.set_fontsize(9)
        label.set_color('#424242')

    # 2. Botones de provincias
    if provincias_dict is not None:
        # Botón para generar todas las provincias
        btn_todas_ax = plt.axes([0.67, 0.60, 0.30, 0.05])
        btn_todas = Button(btn_todas_ax, 'GENERAR TODAS LAS PROVINCIAS', 
                          color='#4CAF50', hovercolor='#45A049')
        
        def generar_todas_callback(event):
            metodo = radio.value_selected
            print("\n" + "="*70)
            print("INICIANDO GENERACIÓN MASIVA DE PROVINCIAS")
            print("="*70)
            generar_todas_provincias(provincias_dict, metodo)
        
        btn_todas.on_clicked(generar_todas_callback)
        
        # Lista de provincias para selección individual
        provincias_lista = sorted(provincias_dict['provincia'].unique())
        
        # Crear área de scroll para provincias (simulado con texto)
        info_prov_ax = plt.axes([0.67, 0.25, 0.30, 0.30], facecolor='#FFF3E0')
        info_prov_ax.axis('off')
        
        provincias_texto = "PROVINCIAS DISPONIBLES:\n" + "─"*35 + "\n"
        for i, prov in enumerate(provincias_lista, 1):
            provincias_texto += f"{i:2d}. {prov}\n"
        
        info_prov_ax.text(
            0.05, 0.98, provincias_texto,
            ha='left', va='top',
            fontsize=7,
            color='#E65100',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF9800', linewidth=1)
        )
        
        # Campo de texto para ingresar número de provincia
        txt_provincia_ax = plt.axes([0.67, 0.18, 0.20, 0.04])
        txt_provincia = TextBox(txt_provincia_ax, 'N° Provincia:', initial='')
        
        # Botón para generar provincia individual desde interfaz
        btn_individual_ax = plt.axes([0.67, 0.13, 0.30, 0.04])
        btn_individual = Button(btn_individual_ax, 'Generar Provincia por Número', 
                               color='#2196F3', hovercolor='#1976D2')
        
        def generar_individual_callback(event):
            metodo = radio.value_selected
            numero_texto = txt_provincia.text.strip()
            
            if not numero_texto:
                print("\n⚠ Por favor ingresa un número de provincia en el campo de texto.")
                return
            
            try:
                numero = int(numero_texto)
                if 1 <= numero <= len(provincias_lista):
                    provincia_seleccionada = provincias_lista[numero - 1]
                    print(f"\n{'='*70}")
                    print(f"GENERANDO MODELO PARA: {provincia_seleccionada} (#{numero})")
                    print(f"{'='*70}")
                    generar_stl_provincia(provincia_seleccionada, provincias_dict, metodo)
                    
                    # Limpiar el campo de texto
                    txt_provincia.set_val('')
                else:
                    print(f"\n⚠ Número inválido. Debe estar entre 1 y {len(provincias_lista)}")
            except ValueError:
                print(f"\n⚠ '{numero_texto}' no es un número válido. Ingresa un número del 1 al {len(provincias_lista)}")
        
        btn_individual.on_clicked(generar_individual_callback)
    
    # Agregar cuadro de información
    info_ax = plt.axes([0.67, 0.02, 0.30, 0.10], facecolor='#E3F2FD')
    info_ax.axis('off')
    info_text = (
        "MODO MANUAL:\n"
        "1. Selecciona algoritmo\n"
        "2. Dibuja rectángulo en el mapa\n"
        "3. STL se guarda automáticamente"
    )
    info_ax.text(
        0.5, 0.5, info_text,
        ha='center', va='center',
        fontsize=8,
        color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#1976D2', linewidth=2)
    )

    # Conexión de eventos de interacción del usuario
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Configuración del selector de rectángulo con estilo mejorado
    selector = RectangleSelector(
        ax, 
        al_seleccionar, 
        useblit=True, 
        button=[1], 
        interactive=True,
        props=dict(facecolor='#1976D2', alpha=0.3, edgecolor='#0D47A1', linewidth=2)
    )
    
    print("\n" + "="*70)
    print("✓ INTERFAZ LISTA")
    print("="*70)
    if provincias_dict is not None:
        print(f"✓ {len(provincias_dict)} provincias cargadas y visualizadas en el mapa")
        print("  - Usa 'GENERAR TODAS LAS PROVINCIAS' para procesamiento masivo")
        print("  - O dibuja un rectángulo para selección manual")
    else:
        print("  - Dibuja un rectángulo para selección manual")
    print("  - Los archivos STL se guardarán en 'modelos_stl/'")
    print("  - Usa la rueda del ratón para hacer zoom")
    print("="*70 + "\n")
    plt.show()

except Exception as e:
    print(f"Error crítico en la interfaz: {e}")