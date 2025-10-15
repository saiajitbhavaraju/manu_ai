delete qdrant_db
enter api key in backend/.env

to test backend:
run 
`python ingest.py
uvicorn main:app --reload --port 8001`
post this curl command in a terminal: 
`curl -X POST http://127.0.0.1:8001/query -H "Content-Type: application/json" -d "{\"question\": \"What is risk management?\"}"`

to test frontend, run the backend using curl, then
`npm install
npm start`
