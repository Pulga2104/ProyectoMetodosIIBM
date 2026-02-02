import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, RadioButtons
import rasterio
import numpy as np
from stl import mesh
from scipy.ndimage import uniform_filter, median_filter
import os
import datetime
import sys

# Importación del módulo personalizado de procesamiento
from src.procesador import verificar_o_crear_mapa

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
CARPETA_DATOS = "datos_srtm"      
ARCHIVO_FINAL = "ecuador_completo.tif"
CARPETA_STL = "modelos_stl"

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
    fig = plt.figure(figsize=(14, 10), facecolor='#F5F5F5')
    
    # Ajuste de márgenes para mejor distribución
    ax = plt.axes([0.08, 0.1, 0.70, 0.85])
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

    # Configuración de título y etiquetas con estilo profesional
    ax.set_title(
        "MAPA INTERACTIVO 3D - Generador de Modelos Topográficos\n"
        "Usa la rueda del ratón para hacer zoom. Dibuja un rectángulo para seleccionar el área a exportar.", 
        fontsize=13, 
        fontweight='bold',
        color='#1976D2',
        pad=15
    )
    ax.set_xlabel("Longitud", fontsize=11, fontweight='bold', color='#424242')
    ax.set_ylabel("Latitud", fontsize=11, fontweight='bold', color='#424242')
    
    # Mejorar apariencia de los ticks
    ax.tick_params(colors='#424242', labelsize=10)

    # Creación del panel de controles con diseño mejorado
    ejes_botones = plt.axes([0.80, 0.40, 0.18, 0.25], facecolor='#FAFAFA')
    ejes_botones.set_title(
        'Algoritmo de Suavizado', 
        fontsize=11, 
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
        label.set_fontsize(10)
        label.set_color('#424242')

    # Agregar cuadro de información
    info_ax = plt.axes([0.80, 0.15, 0.18, 0.20], facecolor='#E3F2FD')
    info_ax.axis('off')
    info_text = (
        "INSTRUCCIONES:\n\n"
        "1. Haz zoom con la rueda\n"
        "2. Selecciona un algoritmo\n"
        "3. Dibuja un rectángulo\n"
        "   sobre el área deseada\n"
        "4. El modelo STL se\n"
        "   guardará automáticamente"
    )
    info_ax.text(
        0.5, 0.5, info_text,
        ha='center', va='center',
        fontsize=9,
        color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#1976D2', linewidth=2)
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
    
    print("Interfaz lista. Los archivos STL se guardarán en la carpeta 'modelos_stl'.")
    print("Haz zoom para ver más detalles en el mapa de calles.")
    plt.show()

except Exception as e:
    print(f"Error crítico en la interfaz: {e}")