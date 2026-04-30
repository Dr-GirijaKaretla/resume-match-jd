FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects GEMINI_API_KEY and PORT automatically
ENV GEMINI_API_KEY=""
ENV PORT=8501

EXPOSE 8501

CMD ["./start.sh"]
