FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Créez les dossiers nécessaires
RUN mkdir -p static/images

# Exposez le port sur lequel Flask s'exécutera
EXPOSE 8080

# Commande pour exécuter l'application
CMD ["python", "app.py"]