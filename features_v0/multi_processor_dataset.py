from analysis_functions import *
import multiprocessing as mp
from functools import partial

def process_single_suffix(suffix_data):
    """
    Procesa un único suffix.
    
    Args:
        suffix_data (tuple): (suffix, cell_ids, data_dir, features_dir, ids_sat, normalize, fs, number_peaks, base_counter)
    
    Returns:
        tuple: (processed_features, errors, suffix)
    """
    # Extraer datos del tuple
    suffix, cell_ids, data_dir, features_dir, ids_sat, normalize, fs, number_peaks, base_counter = suffix_data
    
    error_list = []
    processed_features = 0
    local_feature_counter = base_counter  # Contador local para este proceso
    
    print(f"\nProcesando suffix {suffix}...")
    try:
        # Cargar muestras y datos
        samples = load_samples(data_dir, suffix)
        ra_sat, ra_cell = load_data(data_dir, suffix)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'ra_sat': ra_sat,
            'ra_cell': ra_cell,
            'samples': list(samples)  # Almacenar señales en listas
        })
        
        # Índice para búsqueda más rápida
        df.set_index(['ra_sat', 'ra_cell'], inplace=True)
        
        # Procesando todas las celdas para este suffix
        for cell in cell_ids:
            # Crear directorio específico para la celda
            cell_dir = os.path.join(features_dir, f"cell_{cell}")
            os.makedirs(cell_dir, exist_ok=True)
            
            print(f"\nProcesando celda {cell} para suffix {suffix}...")
            signals_cell = get_signals(df=df, cell_id=cell)
            
            if signals_cell:
                # Procesar señales para esta celda
                sat_count = 0
                error_count = 0
                
                for idx, (sat_id, signal) in enumerate(signals_cell):
                    if sat_id in ids_sat:
                        try:
                            print(f"  - Celda {cell}, Satélite {sat_id} (índice {idx}): Forma de señal {signal.shape}")
                            
                            # Extraer componentes I/Q
                            i_signal = signal[:, 0]
                            q_signal = signal[:, 1]
                            
                            # Calcular características
                            features_mag, features_phase = iq_features_processor_v1(
                                i_signal, q_signal, 
                                fs=fs, 
                                number_peaks=number_peaks, 
                                normalize=normalize
                            )
                            
                            # Incrementar contador local
                            current_counter = local_feature_counter
                            local_feature_counter += 1
                            
                            # Crear matriz de características
                            features, metadata = create_feature_matrix(
                                features_mag, features_phase, 
                                sat_label=sat_id, 
                                transmitter_label=cell, 
                                feature_counter=current_counter, 
                                normalize=normalize
                            )
                            
                            processed_features += 1
                            sat_count += 1
                            
                            # Guardar archivos en el directorio específico de la celda
                            feature_filename = f"{metadata['Feature_Matrix_Name']}"
                            # Añadir suffix al nombre para evitar colisiones
                            safe_filename = f"{feature_filename}_suffix_{suffix}"
                            
                            np.save(os.path.join(cell_dir, f"{safe_filename}.npy"), features)
                            with open(os.path.join(cell_dir, f"{safe_filename}_metadata.json"), "w") as f:
                                json.dump(metadata, f, indent=2)
                                
                        except Exception as e:
                            error_count += 1
                            error_details = {
                                "suffix": suffix,
                                "cell_id": cell,
                                "sat_id": sat_id,
                                "df_index": idx,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                            error_list.append(error_details)
                            
                            print(f"  ERROR procesando señal: Suffix {suffix}, Celda {cell}, Satélite {sat_id}, Índice {idx}")
                            print(f"  Detalles del error: {str(e)}")
                            print("  Continuando con la siguiente señal...")
                
                print(f"  Procesados {sat_count} satélites para la celda {cell}, encontrados {error_count} errores")
            else:
                print(f"  No se encontraron señales para la celda {cell}")
        
    except Exception as e:
        print(f"Error procesando el suffix completo {suffix}: {str(e)}")
        error_details = {
            "suffix": suffix,
            "cell_id": "ALL",
            "sat_id": "ALL",
            "df_index": "N/A",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        error_list.append(error_details)
    
    return (processed_features, error_list, suffix)

def process_cell_features_parallel(max_suffix=171, cell_ids=None, num_cores=None):
    """
    Procesa características de celdas con parámetros configurables y manejo detallado de errores,
    utilizando procesamiento paralelo.
    
    Args:
        max_suffix (int): Número máximo de suffix a procesar (por defecto: 171)
        cell_ids (list): Lista de IDs de celdas a procesar (por defecto: [1, 2, 3])
        num_cores (int): Número de núcleos a utilizar (por defecto: todos los disponibles)
    """
    if cell_ids is None:
        cell_ids = list(range(1, 3))  # Por defecto celdas 1-3
    
    if num_cores is None:
        num_cores = mp.cpu_count()  # Usar todos los núcleos disponibles
    else:
        num_cores = min(num_cores, mp.cpu_count())  # No usar más núcleos de los disponibles
    
    print(f"Utilizando {num_cores} núcleos para el procesamiento paralelo")
    
    #data_dir = "../../data/"
    #data_dir = "/home/carlos/Documents/fingerprint/data"
    data_dir = "/home/carlos/fingerprint/data"
    features_dir = "../../features/"  # Directorio base para características
    
    # Generar suffixes basados en el parámetro de entrada
    suffixes = [f"{i:03d}" for i in range(max_suffix)]
    
    normalize = False
    fs = 25e6           # Frecuencia de muestreo
    number_peaks = 10   # Número de picos
    
    ids_sat = [2, 3, 4, 5, 6, 7, 8, 9, 13, 16,
         17, 18, 22, 23, 24, 25, 26, 28, 29, 30,
         33, 36, 38, 39, 40, 42, 43, 44, 46, 48,
         49, 50, 51, 57, 65, 67, 68, 69, 71, 72,
         73, 74, 77, 78, 79, 81, 82, 85, 87, 88,
         89, 90, 92, 93, 94, 96, 99, 103, 104, 107,
         109, 110, 111, 112, 114, 115]
    
    # Asegurar que el directorio base de características existe
    os.makedirs(features_dir, exist_ok=True)
    
    # Preparar los datos para cada proceso
    # Asignamos un rango de contadores para cada suffix para evitar colisiones
    base_range = 100000  # Un rango amplio para evitar colisiones
    process_data = []
    
    for i, suffix in enumerate(suffixes):
        # Asignar un rango único de contadores para cada suffix
        base_counter = 1 + (i * base_range)
        
        # Preparar datos para este suffix
        suffix_data = (
            suffix, 
            cell_ids, 
            data_dir, 
            features_dir, 
            ids_sat, 
            normalize, 
            fs, 
            number_peaks, 
            base_counter
        )
        process_data.append(suffix_data)
    
    print(f"Iniciando procesamiento para {len(cell_ids)} celdas: {cell_ids}")
    print(f"Procesando {len(suffixes)} suffixes (0 a {max_suffix-1})")
    
    # Crear archivo de informe de errores
    error_report_path = os.path.join(features_dir, "error_report.txt")
    
    # Crear un pool de procesos
    with mp.Pool(processes=num_cores) as pool:
        # Ejecutar procesos en paralelo
        results = pool.map(process_single_suffix, process_data)
    
    # Procesar resultados
    total_features = 0
    all_errors = []
    
    for feat_count, errors, suffix in results:
        print(f"Suffix {suffix} completado: {feat_count} características generadas, {len(errors)} errores")
        total_features += feat_count
        all_errors.extend(errors)
    
    # Escribir informe de errores
    with open(error_report_path, "w") as error_file:
        error_file.write(f"INFORME DE ERRORES - Generado el {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        error_file.write(f"Total de errores encontrados: {len(all_errors)}\n\n")
        
        if len(all_errors) > 0:
            for i, error in enumerate(all_errors, 1):
                error_file.write(f"ERROR #{i}\n")
                error_file.write(f"Suffix: {error['suffix']}\n")
                error_file.write(f"ID de Celda: {error['cell_id']}\n")
                error_file.write(f"ID de Satélite: {error['sat_id']}\n")
                error_file.write(f"Índice de DataFrame: {error['df_index']}\n")
                error_file.write(f"Mensaje de Error: {error['error']}\n")
                error_file.write(f"Traceback:\n{error['traceback']}\n")
                error_file.write("-" * 80 + "\n\n")
    
    print(f"\nProcesamiento completo:")
    print(f"- Total de características generadas: {total_features}")
    print(f"- Total de errores encontrados: {len(all_errors)}")
    print(f"- Informe de errores guardado en: {error_report_path}")
    
    # Paso opcional: renumerar de forma secuencial los archivos
    print("\nPaso opcional: Realizar renumeración secuencial de los archivos...")
    renumber_feature_files(features_dir, cell_ids)

def renumber_feature_files(features_dir, cell_ids):
    """
    Renumera los archivos de características para que tengan números secuenciales.
    Esto es útil después del procesamiento paralelo.
    
    Args:
        features_dir (str): Directorio base de características
        cell_ids (list): Lista de IDs de celdas procesadas
    """
    counter = 1
    
    for cell in cell_ids:
        cell_dir = os.path.join(features_dir, f"cell_{cell}")
        if not os.path.exists(cell_dir):
            continue
        
        # Obtener todos los archivos .npy que no son de metadatos
        npy_files = [f for f in os.listdir(cell_dir) if f.endswith('.npy') and not f.endswith('_metadata.json')]
        
        # Procesar cada archivo
        for old_npy in npy_files:
            # Cargar datos y metadata
            data_path = os.path.join(cell_dir, old_npy)
            metadata_path = os.path.join(cell_dir, old_npy.replace('.npy', '_metadata.json'))
            
            if not os.path.exists(metadata_path):
                print(f"No se encontró metadata para {data_path}, omitiendo...")
                continue
            
            try:
                # Cargar datos
                features = np.load(data_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Crear nuevo nombre de archivo con contador secuencial
                sat_id = metadata.get('Satellite_Label', '0')
                cell_id = metadata.get('Transmitter_Label', '0')
                new_name = f"Feature_Matrix_{counter:06d}_sat_{sat_id}_tx_{cell_id}"
                
                # Actualizar metadata
                metadata['Feature_Matrix_Name'] = new_name
                metadata['Feature_Counter'] = counter
                
                # Guardar con nuevo nombre
                new_data_path = os.path.join(cell_dir, f"{new_name}.npy")
                new_metadata_path = os.path.join(cell_dir, f"{new_name}_metadata.json")
                
                np.save(new_data_path, features)
                with open(new_metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Eliminar archivos antiguos una vez que los nuevos están guardados
                os.remove(data_path)
                os.remove(metadata_path)
                
                counter += 1
                
            except Exception as e:
                print(f"Error al renumerar {data_path}: {str(e)}")
    
    print(f"Renumeración completada. Total de {counter-1} archivos procesados.")

if __name__ == "__main__":
    # Ejemplo de cómo llamar a la función con parámetros personalizados
    max_suffix = 10  # Procesar suffixes 000 a 170
    cell_ids = list(range(35, 45))  # Procesar celdas de la 35 a la 44
    num_cores = None  # Usar 15 núcleos

    
    start_time = time.time()
    process_cell_features_parallel(max_suffix=max_suffix, cell_ids=cell_ids, num_cores=num_cores)
    elapsed_time = time.time() - start_time
    print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")