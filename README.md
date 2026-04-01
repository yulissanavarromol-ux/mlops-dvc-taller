# MLOps DVC Base

Repositorio base para el Taller de DVC - BCD4205

## Estructura

```
mlops-dvc-base/
├── src/
│   └── train.py
├── pyproject.toml
└── README.md
```

## Instrucciones

Sigue las instrucciones del taller para:

1. Inicializar DVC
2. Descargar y versionar el dataset
3. Entrenar y versionar el modelo
4. Configurar Google Drive como remote
5. Sincronizar con tu repositorio

## Instalación

```bash
uv sync
```

## Ejecución

```bash
uv run python src/train.py
```

# Comentarios importantes:

```bash
# Configurar AWS CLI
aws configure --profile nombre-del-perfil

# Agregar remote de S3
dvc remote add -d myremote s3://tu-bucket/ruta/dvc-storage

# Si usas un perfil específico de AWS
dvc remote modify myremote profile nombre-del-perfil
```