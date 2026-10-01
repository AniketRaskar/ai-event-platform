# Project: AI Event Platform v2

A college event platform with (1) a hybrid Neural Collaborative Filtering recommender (NeuMF + SBERT content embeddings) and (2) a RAG chatbot that answers questions about events and can call the recommender as a tool. Everything must run locally with: SQLite, no API keys required, no Docker required.

## Hard rules
- Python 3.10+, type hints, small functions, docstrings on public functions.
- Never fabricate results. All metrics shown anywhere must come from scripts that actually ran. If a result file does not exist, the README must say "run `make eval`".
- All randomness seeded (SEED=42 in config).
- Synthetic data must be clearly labeled as synthetic in README and in generated files.
- Every module in `ml/` and `rag/` is a plain importable package with no FastAPI dependency. The backend imports them.
- Each phase ends with a runnable command and at least one test. Do not move on if tests fail.
- Keep dependencies minimal. Prefer standard library and the packages named here.

## Stack
FastAPI, SQLAlchemy 2.x, Pydantic v2, SQLite, PyTorch, sentence-transformers (all-MiniLM-L6-v2), scikit-learn, rank_bm25, numpy, pandas, pytest, React + Vite.

## Database tables
- users(id, name, email unique, password_hash, role['student'|'admin'], department, year, interests JSON list, created_at)
- friendships(user_id, friend_id)  # symmetric pairs stored both directions
- events(id, title, description, category, department, organizer, venue, start_time, end_time, registration_deadline, fee, capacity, eligibility, rules, faq JSON list of {q,a}, posted_at, created_by)
- interactions(id, user_id, event_id, type['view'|'like'|'register'|'dislike'|'feedback'], rating nullable, comment nullable, timestamp)
- notifications(id, user_id, event_id, message, created_at, read)

Positive signals for training/eval: like, register. 'view' is weak and excluded from positives. 'dislike' is a hard negative.

## Synthetic data spec (ml/data/synthetic_generator.py)
- 500 users, 250 events, 8 categories: AI/ML, Web/Software, Cybersecurity, Robotics/IoT, Sports, Cultural, Entrepreneurship, Higher Studies/Career.
- 5 departments. Each user has 1-3 favourite categories (interest clusters) and a department bias.
- 20 friend groups of 15-30 users; each user also has 0-3 random cross-group friends. Friends share interests with probability 0.6.
- Events posted across 12 months (posted_at), start_time 7-30 days after posted_at. Organizers are 15 clubs with varying sizes (some clubs small, to enable a later fairness check).
- Popularity skew: event base popularity follows a power law.
- Interactions: for each user, over each event's open window (posted_at -> start_time), generate view/like/register with probability depending on category match, department match, popularity, and number of friends who already registered. Timestamps fall in the open window. About 25 percent of 'like' events also get 'register'.
- Event text (title, description, rules, eligibility, faq) comes from templates in `event_docs.py`, varied by category with randomised details (names, dates, prizes, fees, venues). Make the text rich enough for RAG Q&A: eligibility, team size, fee, deadline, prizes, venue, contact, rules, 3-5 FAQ items.
- Output: CSVs in `ml/data/generated/` plus `seed_db.py` to load into SQLite. Create a demo admin (admin@college.edu / admin123) and a demo student (student@college.edu / student123).

## Split and evaluation spec
- Time-based split on interaction timestamp: train = first 70 percent of the timeline, val = next 10 percent, test = last 20 percent.
- Event flags per split: an event is WARM if it has >=1 positive interaction in train, otherwise COLD (this includes every event posted after the train cutoff).
- Test candidates for a user = events open (posted_at <= t <= start_time) at the user's test interaction time, excluding events the user already positively interacted with in train. Rank all candidates (no negative sampling at eval).
- Metrics: Recall@10, NDCG@10, HitRate@10, catalog coverage@10. Report on ALL, WARM-only targets, COLD-only targets. Write results to `docs/results/results.json` and `results.md`.
- metrics.py must be unit tested on a hand-made tiny example with known answers.

## Models (ml/models)
1. PopularityBaseline: count of positives in train, recency weighted.
2. TfidfKnnBaseline: TF-IDF of event text; user profile = mean of vectors of positively-interacted events; score by cosine similarity (this mirrors the original published paper).
3. MFBaseline: PyTorch matrix factorization, BCE loss, 4 negatives per positive.
4. NeuMF (no content): user and event ID embeddings; GMF branch = elementwise product; MLP branch = concat -> 64 -> 32 -> 16 (ReLU, dropout 0.2); concat both branch outputs -> linear -> sigmoid. BCE loss, 4 sampled negatives per positive, Adam.
5. NeuMF + content (the headline model): event representation = [event ID embedding ; Linear(384 -> d) of the frozen SBERT description embedding]. During training, ID DROPOUT: with p=0.3 replace the event ID embedding with a learned UNK vector so the model learns to score from content alone. At inference, an event unseen in training always uses UNK. Users also get a side-feature vector (multi-hot interests + department one-hot) projected and concatenated with the user ID embedding.
6. Ablations (ablations.py): no GMF branch, no MLP branch, no content, id-dropout 0 vs 0.3 vs 0.5, negatives 1/4/8.
- Early stop on validation NDCG@10. Save the final model, id maps and embedding cache into `ml/artifacts/`.

## RAG spec (rag/)
- Corpus: one chunk-set per event (header chunk with title/date/venue/fee/category/eligibility; rules chunk; each FAQ item as its own chunk), plus markdown files in `rag/data/general_docs/` (hostel rules, attendance policy, how to register, club contacts). Every chunk carries metadata: event_id, category, department, start_time, fee, doc_type.
- Retrieval: BM25 (rank_bm25) + dense (SBERT cosine in NumPy), merged by Reciprocal Rank Fusion (k=60). Optional cross-encoder rerank behind a config flag (default off).
- Query rewriting (rule-based, no LLM needed): "today", "tomorrow", "this weekend", "next week", "free" (fee=0), "paid", department names, category keywords -> metadata filters. Fall back to no filter if filters return nothing.
- Generation: build a prompt with numbered context chunks; the model must answer ONLY from context and cite like [E12]. If the top retrieval score is below a threshold, or no chunks pass filters, return "I don't have that information" without calling the LLM.
- LLM interface `llm.py`: `generate(prompt) -> str` with providers selected by env `LLM_PROVIDER` = `ollama` | `openai_compatible` | `extractive`. `extractive` is the default offline fallback: returns the best-matching chunk sentences with citations, so the project works with zero setup.
- Router: rule-based first (keywords like "suggest", "recommend", "what should I attend", "for me" -> RECOMMEND; otherwise INFO), with an optional LLM classifier behind a flag. RECOMMEND calls `tools.recommend_events(user_id, filters)` and presents results with reasons.
- Index freshness: `ingest.add_event(event)` updates the BM25 and dense indexes in memory and on disk when an admin posts an event.
- Eval: `build_testset.py` auto-generates about 50 questions from event fields (fee, deadline, venue, eligibility, team size) with gold event ids, plus 10 unanswerable questions. `run_eval.py` reports hit@3 per retriever (BM25-only, dense-only, hybrid), refusal accuracy, and mean latency, saved to `docs/results/rag_results.json`.

## Backend API
- POST /auth/register, POST /auth/login (JWT), GET /auth/me, PUT /auth/me/interests
- GET /events (filters: category, department, free, q, upcoming), GET /events/{id}
- POST /events (admin), PUT /events/{id} (admin)  -> on create: embed, add to RAG index, create notifications for users whose recommender score is in the top 20 percent
- POST /events/{id}/register, POST /events/{id}/like, POST /events/{id}/dislike, POST /events/{id}/view, POST /events/{id}/feedback
- GET /recommend?k=10 -> list of events with score and a short `reason` (for example "Matches your interest in AI/ML; 3 friends registered")
- POST /chat {message, history} -> {answer, citations[], route, used_tool}
- GET /admin/events/{id}/registrations, GET /admin/analytics (registrations per event, likes per category, top events, recommendation click-through placeholders)
- GET /notifications, POST /notifications/{id}/read
- New users with no history: cold-start path = selected interests (content similarity) + popularity.
- Log every /recommend impression and every click (view) for future analysis.

## Frontend (React + Vite)
Pages: Login, Register, Onboarding (pick interest chips), Feed ("Recommended for you" with reason text), Events (search + filters), EventDetail (register, like, feedback), Admin (create event form, registrations list, likes, analytics charts as simple tables or bars), floating ChatWidget on every page that shows answers with clickable citations. Plain CSS, clean and responsive. API base URL from `VITE_API_URL`.

## Makefile targets
setup, data (generate + seed), train, eval (all models + ablations), index (build RAG index), rag-eval, api (uvicorn on 8000), web (vite on 5173), test.

## Definition of done
`make setup && make data && make train && make index && make api` plus `make web` gives a working app at localhost:5173 with login, personalised feed, chatbot answering with citations, and admin posting an event that the chatbot knows about immediately. `make eval` and `make rag-eval` write result tables to `docs/results/`.

