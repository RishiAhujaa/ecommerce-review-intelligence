# Running the Dashboard

## Start
```bash
cd ~/BTP_Project
source venv/bin/activate
streamlit run app.py
```

## Stop
```bash
pkill -f "streamlit run app.py"
```

## Restart
```bash
pkill -f "streamlit run app.py" && sleep 1 && source venv/bin/activate && streamlit run app.py
```

## URL
```
http://localhost:8501
```
