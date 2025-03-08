# shyftlabs-simple-rag
### RUN
*Sorry forgot to clean my requirements.txt (re-used env from a previous project)
- You'll need your OpenAI API key in an .env file
- use test.py
### Some notes to the reviewer about my thought process:
- Only considering the boundry that 2 filetypes will be accepted.
- Since this program will probably only be run once, chose a Non-Persistent DB
you could practically say that it is a mock db.
- chose to save the uploaded files just in case, not necessary however.
- also attached the original DeepSeek paper metioned in the instructions.
### IDK how further in I was supposed to get, but here are some thoughts for features down the road:
- Implement Summarization, depending on constraints regarding each document accepted.
- Faiss Algo if doc type is an important constraint.
- Vectorize perhaps differently, 2 versions --> by n-grams besides semantic split.
- Implement real vec DB
- use different prompt enhancer --> rank a couple of prompts.
- Used some v0.1 features from Langchain, move to v0.3 for more maintainability.
- Is FastAPI the best for this?
- Containerize (Docker?)



