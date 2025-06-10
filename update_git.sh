#!/usr/bin/env bash
#
# update_git.sh — automatiza checkout, pull, add, commit y push en 'main'
#

set -euo pipefail

# Ajusta si usas otra rama principal
BRANCH="main"

# 1. Cambiar a main
echo "» Cambiando a la rama $BRANCH…"
git checkout "$BRANCH"

# 2. Traer cambios del remoto
echo "» Haciendo pull origin/$BRANCH…"
git pull origin "$BRANCH"

# 3. (Trabajo manual aquí)

# 4. Añadir todos los cambios
echo "» Añadiendo archivos…"
git add .

# 5. Commit (mensaje pasado como primer parámetro)
if [ $# -eq 0 ]; then
  echo "Error: necesitas pasar un mensaje de commit."
  echo "Uso: $0 \"Mensaje descriptivo de tus cambios\""
  exit 1
fi
echo "» Haciendo commit: \"$1\""
git commit -m "$1"

# 6. Push al remoto
echo "» Haciendo push origin/$BRANCH…"
git push origin "$BRANCH"

echo "¡Listo! Tu rama $BRANCH está actualizada y tus cambios han sido subidos."
