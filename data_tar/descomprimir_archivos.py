import os
import tarfile
import shutil
from tqdm import tqdm

def extraer_archivos_targz(directorio_origen, directorio_destino, extraer_todo_junto=True):
    """
    Descomprime archivos .tar.gz con formato data_XXX_XXX.tar.gz y guarda su contenido
    en un directorio específico. Ofrece la opción de borrar los archivos comprimidos tras la extracción.
    
    Args:
        directorio_origen (str): Ruta al directorio donde están los archivos .tar.gz
        directorio_destino (str): Ruta al directorio donde se guardarán los archivos descomprimidos
        extraer_todo_junto (bool): Si es True, extrae todos los archivos en el mismo directorio.
                                  Si es False, crea un subdirectorio para cada archivo comprimido.
    """
    # Crear el directorio de destino si no existe
    os.makedirs(directorio_destino, exist_ok=True)
    
    # Obtener lista de archivos .tar.gz en el directorio de origen
    archivos_targz = [archivo for archivo in os.listdir(directorio_origen) 
                     if archivo.endswith('.tar.gz') and archivo.startswith('data_')]
    
    if not archivos_targz:
        print(f"No se encontraron archivos .tar.gz en {directorio_origen}")
        return
    
    print(f"Se encontraron {len(archivos_targz)} archivos .tar.gz para extraer")
    
    # Preguntar al usuario si desea eliminar los archivos comprimidos después de extraerlos
    while True:
        respuesta = input("¿Deseas borrar los archivos comprimidos después de extraerlos? (yes/no): ").lower()
        if respuesta in ['yes', 'no']:
            borrar_comprimidos = (respuesta == 'yes')
            break
        else:
            print("Por favor, responde 'yes' o 'no'")
    
    # Ordenar los archivos por su número de secuencia
    archivos_targz.sort(key=lambda x: int(x.split('_')[1]))
    
    archivos_exitosos = 0
    
    # Extraer cada archivo .tar.gz
    for archivo in tqdm(archivos_targz, desc="Extrayendo archivos"):
        ruta_completa = os.path.join(directorio_origen, archivo)
        extraccion_exitosa = False
        
        try:
            # Determinar el directorio de destino para la extracción
            if extraer_todo_junto:
                dir_extraccion = directorio_destino
            else:
                # Crear un subdirectorio basado en el nombre del archivo sin la extensión
                nombre_base = os.path.splitext(os.path.splitext(archivo)[0])[0]  # Elimina .tar.gz
                dir_extraccion = os.path.join(directorio_destino, nombre_base)
                os.makedirs(dir_extraccion, exist_ok=True)
            
            # Extraer el archivo .tar.gz
            with tarfile.open(ruta_completa, 'r:gz') as tar:
                tar.extractall(path=dir_extraccion)
            
            extraccion_exitosa = True
            archivos_exitosos += 1
                
        except Exception as e:
            print(f"\nError al extraer {archivo}: {e}")
            extraccion_exitosa = False
        
        # Borrar el archivo comprimido si la extracción fue exitosa y el usuario lo solicitó
        if extraccion_exitosa and borrar_comprimidos:
            try:
                os.remove(ruta_completa)
                print(f"\nArchivo {archivo} eliminado correctamente")
            except Exception as e:
                print(f"\nNo se pudo eliminar el archivo {archivo}: {e}")
    
    print(f"\nProceso de extracción completado. {archivos_exitosos} de {len(archivos_targz)} archivos fueron extraídos correctamente.")
    print(f"Los archivos descomprimidos están en {directorio_destino}")
    if borrar_comprimidos:
        print(f"Los archivos comprimidos extraídos exitosamente han sido eliminados.")

if __name__ == "__main__":
    # Directorios de origen y destino
    directorio_origen = "./zenodo_dataset"  # Donde están los archivos descargados
    directorio_destino = "./datos_extraidos"  # Donde se guardarán los archivos extraídos
    
    # Si quieres extraer todos los archivos en un solo directorio, cambia a True
    extraer_todo_junto = False
    
    # Ejecutar la función de extracción
    extraer_archivos_targz(directorio_origen, directorio_destino, extraer_todo_junto)