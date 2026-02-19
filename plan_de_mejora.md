# Plan de Mejora para la IA de Trellis (Proyecto KubikAI)

Este documento resume el trabajo realizado y los próximos pasos para mejorar el modelo de generación 3D de Trellis.

## Objetivo
Mejorar significativamente la calidad de los modelos 3D generados, enfocándonos en obtener geometrías más nítidas y una mayor fidelidad a los detalles y texturas de la imagen de entrada.

## Análisis y Estrategia
El análisis del código de Trellis reveló una arquitectura de dos etapas. Identificamos dos áreas principales de mejora:

1.  **Geometría del Modelo:** El sistema original usa vóxeles, lo que resulta en formas 3D "redondeadas" y con poco detalle.
2.  **Fidelidad a la Imagen:** El modelo capta la idea general de la imagen, pero pierde detalles finos debido a un mecanismo de condicionamiento simple.

Nuestra estrategia ataca estos dos puntos con una arquitectura completamente nueva que hemos construido dentro de la carpeta `KubikAI`.

## Trabajo Realizado (Fase de Codificación - COMPLETA)

Hemos implementado toda la base de código para las siguientes mejoras:

1.  **Nuevo VAE para Geometría Precisa (SDF-VAE):**
    *   **Acción:** Reemplazamos los vóxeles por **Funciones de Distancia Signada (SDF)**.
    *   **Código Creado:**
        *   `KubikAI/datasets/sdf_dataset.py`: Dataset para nuestros nuevos datos SDF.
        *   `KubikAI/models/sdf_vae.py`: El modelo VAE basado en SDF.
        *   `KubikAI/trainers/sdf_vae_trainer.py`: El entrenador para el `SdfVAE`.
        *   `KubikAI/configs/kubikai_sdf_vae_v1.json`: Configuración para entrenar este VAE.

2.  **Nuevo Modelo de Flujo para Mayor Detalle (Cross-Attention):**
    *   **Acción:** Implementamos **atención cruzada** para que el modelo pueda fijarse en detalles específicos de la imagen.
    *   **Código Creado:**
        *   `KubikAI/models/cross_attention_flow.py`: El nuevo modelo de flujo con atención cruzada.
        *   `KubikAI/trainers/cross_attention_trainer.py`: El entrenador adaptado para este nuevo modelo.
        *   `KubikAI/configs/kubikai_v1.json`: La configuración principal del experimento, actualizada para usar la nueva arquitectura.

3.  **Integración:**
    *   Se ha modificado `train.py` para que sea más flexible y pueda cargar y entrenar nuestros nuevos módulos desde la carpeta `KubikAI` sin problemas.

## Próximos Pasos (Fase de Datos y Entrenamiento) - COMPLETADO Y REVISADO

La fase de codificación de los modelos está lista. La siguiente fase crítica es la preparación de los datos para el entrenamiento.

1.  **Creación de la Pipeline de Pre-procesamiento (COMPLETADO):**
    *   **Tarea:** Se ha creado un script `KubikAI/preprocess_data.py` para convertir modelos 3D (`.obj`) al formato requerido por la IA: un archivo `.npz` con muestras SDF y una serie de imágenes renderizadas desde múltiples ángulos con sus respectivas matrices de cámara.
    *   **Desarrollo y Depuración:**
        *   El enfoque inicial utilizaba un script de Blender pre-existente. Sin embargo, nos encontramos con una serie de errores críticos y silenciosos relacionados con la instalación y versión de Blender del entorno local.
        *   Para solucionar esto, se implementó un sistema de registro y verificación robusto en `preprocess_data.py`, lo que nos permitió diagnosticar con precisión los fallos internos de Blender.
        *   **Pivote Estratégico:** Tras confirmar que los problemas de Blender eran insuperables desde nuestro código, se tomó la decisión de reemplazar por completo la dependencia de Blender. Se re-implementó toda la lógica de renderizado usando las librerías de Python `pyrender` y `trimesh`.
    *   **Resultado:** Ahora contamos con una pipeline de datos 100% autónoma en Python, más rápida, robusta y que produce resultados consistentes sin depender de software externo.

## Fase de Datos: Depuración y Finalización (COMPLETADO)

Al reanudar el procesamiento de datos, nos encontramos con un error crítico de asignación de memoria (`bad allocation`) durante la generación de SDF en el script `KubikAI/preprocess_data.py`. Esto desencadenó una intensa fase de depuración:

*   **Diagnóstico:** Se identificó que la función `trimesh.proximity.signed_distance` fallaba catastróficamente con mallas de complejidad moderada.
*   **Iteraciones de Solución:** Se exploraron múltiples estrategias (procesamiento por lotes, simplificación de mallas, diferentes APIs de voxelización) que revelaron errores de API y problemas fundamentales en las librerías subyacentes.
*   **Solución Final:** Se re-implementó la generación de SDF desde cero utilizando un enfoque de **voxelización + transformación de distancia de SciPy** (`scipy.ndimage.distance_transform_edt`). Esta solución es algorítmicamente sólida, estable en memoria y no depende de las funciones problemáticas de `trimesh`. Tras corregir varios errores menores de implementación, el script `preprocess_data.py` se volvió completamente funcional.

**Resultado:** Los datasets `Anime` y `Fornite` han sido **procesados exitosamente**.

## Fase de Integración: Estrategia de Dataset Combinado (COMPLETADO)

Para permitir el entrenamiento con múltiples fuentes de datos (nuestros datasets + Objaverse), se ha implementado una estrategia de dataset combinado:

1.  **`KubikAI/datasets/sdf_dataset.py`:** Se ha modificado para aceptar una **lista de directorios** y para escanear de forma inteligente diferentes estructuras de carpetas, haciéndolo compatible tanto con nuestros datasets como con otros de formato plano.
2.  **`train.py`:** Se ha modificado ligeramente para permitir que el argumento `--data_dir` acepte una **cadena de rutas separadas por comas**, que luego se pasa como una lista al nuevo `SdfDataset`.

## Fase de Entrenamiento 1: VAE (COMPLETADO)

*   **Acción:** Se ha entrenado el modelo `SdfVAE` con los datasets combinados.
*   **Proceso:**
    1.  **Depuración del Entorno:** Se encontraron y resolvieron numerosas dependencias faltantes.
    2.  **Conflicto Mayor con `kaolin`:** Se descubrió que la dependencia `kaolin` era incompatible con la versión de PyTorch del entorno.
    3.  **Pivote Estratégico:** Para resolver el conflicto, se creó una **pipeline de entrenamiento completamente nueva y autocontenida** dentro de la carpeta `KubikAI`.
    4.  **Depuración del Modelo:** Se corrigió un **defecto de diseño fundamental** en la arquitectura del `SdfVAE` que causaba un error de memoria irrecuperable (`CUDA out of memory`).
*   **Resultado:** El modelo `SdfVAE` ha sido entrenado exitosamente. Los checkpoints están guardados en `./outputs/sdf_vae_training/ckpts/`.

## Fase 2: Refactorización Final y Portabilidad (COMPLETADO)

*   **Objetivo:** Desacoplar por completo el proyecto del código heredado de `trellis` para garantizar la portabilidad a entornos como Kaggle y eliminar definitivamente el "infierno de dependencias".
*   **Proceso:**
    1.  **Análisis de Dependencias:** Se identificó que el `CrossAttentionFlowModel` y su entrenador seguían dependiendo de `trellis`.
    2.  **Re-implementación de Módulos:** Se copiaron los módulos necesarios (`attention`, `transformer`, etc.) de `trellis` a `KubikAI`, modificándolos para eliminar sub-dependencias problemáticas (`spconv`, `kaolin`).
    3.  **Refactorización de Entrenadores:** Se creó un `flow_trainer.py` autocontenido y se eliminó el código heredado de `trellis`.
    4.  **Creación de la Pipeline de Datos de Flujo:** Se implementaron `encode_dataset.py` y `latent_dataset.py`.
    5.  **Creación de la Pipeline de Entrenamiento de Flujo:** Se crearon el script `train_flow.py` y la configuración `kubikai_flow_v1.json`.
    6.  **Gestión de Dependencias Final:** Se generó un archivo `requirements.txt` limpio y documentado.
*   **Resultado:** El proyecto ahora es **totalmente independiente y portátil**.

## Próximos Pasos: Migración y Entrenamiento Final

El proyecto está listo para ser subido a un repositorio de GitHub y utilizado en un entorno de alto rendimiento como Kaggle.

1.  **Subir el código a GitHub.**
2.  **Configurar el Entorno en Kaggle:**
    *   Crear un notebook y subir los datasets (`processed_datasets` y `encoded_datasets`).
    *   Clonar el repositorio de GitHub.
    *   Instalar las dependencias usando `requirements.txt`.
3.  **Ejecutar el Entrenamiento del Modelo de Flujo:**
    *   **Acción:** Lanzar el entrenamiento del `CrossAttentionFlow`.
    *   **Comando (ejemplo para Kaggle):**
        ```bash
        python KubikAI/train_flow.py --config KubikAI/configs/kubikai_flow_v1.json --output_dir /kaggle/working/flow_training --encoded_dir /kaggle/input/<your-encoded-dataset>/encoded_datasets --processed_dir /kaggle/input/<your-processed-dataset>/processed_datasets
        ```
