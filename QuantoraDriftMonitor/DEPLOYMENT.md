# Quantora - Deployment Guide

## GitHub Deployment

### Quick Setup

1. **Initialize Git Repository (if not already done)**
```bash
git init
git add .
git commit -m "Initial commit: Complete Quantora implementation"
```

2. **Create GitHub Repository**
- Go to https://github.com/new
- Create a new repository named `quantora`
- Do NOT initialize with README, .gitignore, or license (we already have these)

3. **Push to GitHub**
```bash
git remote add origin https://github.com/YOUR_USERNAME/quantora.git
git branch -M main
git push -u origin main
```

### Using Replit GitHub Integration

Alternatively, you can use Replit's built-in GitHub integration:

1. Click on the Git/GitHub icon in the left sidebar
2. Click "Create a new repository"
3. Choose repository name: `quantora`
4. Select visibility (public or private)
5. Click "Create repository and push"

## Production Deployment Options

### Option 1: Deploy on Replit

The application is already configured to run on Replit:

1. The workflow `quantora-api` runs the FastAPI server
2. PostgreSQL database is already connected
3. Port 5000 is configured for web access
4. Auto-restart on file changes enabled

### Option 2: Deploy on Cloud Platform

#### Heroku

1. Create `Procfile`:
```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

2. Deploy:
```bash
heroku create quantora-api
heroku addons:create heroku-postgresql:hobby-dev
git push heroku main
```

#### AWS EC2 / DigitalOcean / Linode

1. Set up Ubuntu 22.04 server
2. Install Python 3.11+
3. Install PostgreSQL
4. Clone repository
5. Install dependencies: `pip install -r requirements.txt`
6. Set environment variables
7. Run with systemd or supervisor

Example systemd service file (`/etc/systemd/system/quantora.service`):
```ini
[Unit]
Description=Quantora API Service
After=network.target

[Service]
Type=notify
User=quantora
WorkingDirectory=/opt/quantora
Environment="DATABASE_URL=postgresql://user:pass@localhost/quantora"
ExecStart=/usr/bin/uvicorn app.main:app --host 0.0.0.0 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY example_usage.py .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://quantora:password@db:5432/quantora
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=quantora
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=quantora
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Deploy:
```bash
docker-compose up -d
```

### Option 3: Deploy on Cloud Run / Lambda

For serverless deployment, you'll need to:

1. Add proper health checks
2. Configure cold start optimization
3. Use managed PostgreSQL (Cloud SQL, RDS, etc.)
4. Add connection pooling

## Environment Variables

Required environment variables for production:

```bash
# Database (Required)
DATABASE_URL=postgresql://user:password@host:port/database

# Optional Configuration
PSI_THRESHOLD=0.25
KS_THRESHOLD=0.2
SHARPE_WINDOW=30
SHARPE_THRESHOLD=-0.5
MIN_SAMPLES_FOR_ANALYSIS=20
DEBUG=False
```

## Production Checklist

### Security
- [ ] Set strong database password
- [ ] Enable HTTPS/TLS
- [ ] Add authentication/authorization middleware
- [ ] Set up rate limiting
- [ ] Configure CORS properly
- [ ] Use secrets manager for sensitive data
- [ ] Enable database SSL mode

### Performance
- [ ] Set up database connection pooling
- [ ] Configure proper worker count (2-4x CPU cores)
- [ ] Add Redis for caching
- [ ] Enable gzip compression
- [ ] Set up CDN for static assets
- [ ] Configure database indexes

### Monitoring
- [ ] Set up application logging (Sentry, LogDNA, etc.)
- [ ] Configure health checks
- [ ] Add performance monitoring (New Relic, DataDog)
- [ ] Set up uptime monitoring
- [ ] Configure alerting for critical errors
- [ ] Add metrics dashboard (Grafana, Prometheus)

### Reliability
- [ ] Configure auto-scaling
- [ ] Set up database backups
- [ ] Implement retry logic for external services
- [ ] Add circuit breakers
- [ ] Configure graceful shutdown
- [ ] Test disaster recovery procedures

## Post-Deployment

### 1. Verify Deployment

```bash
# Check API health
curl https://your-domain.com/health

# Test root endpoint
curl https://your-domain.com/

# Access API docs
open https://your-domain.com/docs
```

### 2. Create First Model

```python
import requests

response = requests.post("https://your-domain.com/models", json={
    "name": "production_model_1",
    "description": "Live trading strategy",
    "model_type": "quantitative",
    "baseline_distribution": {
        "feature_1": [0.1, 0.15, 0.12, 0.18],
        "feature_2": [1.5, 1.6, 1.4, 1.7]
    }
})

print(response.json())
```

### 3. Monitor Performance

- Check application logs regularly
- Monitor database performance
- Track API response times
- Review drift analysis accuracy
- Validate PDF report generation

## Scaling Considerations

### Horizontal Scaling

When you need to handle more requests:

1. **Add Load Balancer** (nginx, HAProxy, AWS ALB)
2. **Scale API Servers** (multiple uvicorn instances)
3. **Use Process Manager** (gunicorn with uvicorn workers)

Example with Gunicorn:
```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:5000 \
  --timeout 120
```

### Vertical Scaling

For computationally intensive operations:

1. Increase CPU/RAM for Bayesian sampling
2. Use faster PostgreSQL instance
3. Add GPU support for Neural SDE (if using TorchSDE)

### Background Processing

For production workloads, consider:

1. **Celery** for background tasks
2. **RabbitMQ** or **Redis** as message broker
3. Separate workers for drift analysis
4. Queue management for PDF generation

## Support

- Documentation: See README.md
- API Reference: https://your-domain.com/docs
- Issues: GitHub Issues

## License

MIT License - See LICENSE file
