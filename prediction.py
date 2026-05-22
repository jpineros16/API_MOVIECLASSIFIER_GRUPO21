#!/usr/bin/python
import re
import os
import sys
import numpy as np
import pandas as pd
import joblib

# ── Géneros en el mismo orden que el MLB ──────────────────────────────────────
GENRE_NAMES = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
    'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
    'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western'
]

# ── Carga del modelo (una sola vez al importar el módulo) ─────────────────────
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'movie_genres.pkl')
_model = joblib.load(_MODEL_PATH)


def _clean_text(text: str) -> str:
    """Misma limpieza usada durante el entrenamiento."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def predict_genres(params: dict) -> dict:
    """
    Recibe un dict con claves 'title', 'plot', 'year' y devuelve
    un dict {genre: probabilidad} para los 24 géneros.
    """
    title = str(params.get('title', ''))
    plot  = str(params.get('plot',  ''))
    year  = int(params.get('year',  0))

    # ── Texto limpio ──────────────────────────────────────────────────────────
    text = _clean_text(title + ' ' + plot)

    # ── Features tabulares (décadas + año escalado) ───────────────────────────
    all_decades   = _model['all_decades']          # lista ordenada de décadas
    year_scaler   = _model['year_scaler']

    decade_val    = (year // 10) * 10
    decade_vec    = np.array(
        [1.0 if d == decade_val else 0.0 for d in all_decades],
        dtype=float
    ).reshape(1, -1)
    year_scaled   = year_scaler.transform([[year]])
    tabular       = np.hstack([decade_vec, year_scaled])   # (1, n_decades+1)

    # ── Vectorización TF-IDF ──────────────────────────────────────────────────
    from scipy import sparse
    vw = _model['vw']
    vc = _model.get('vc')          # puede no existir si se entrenó sin char_wb

    Xw = vw.transform([text])
    parts = [Xw]
    if vc is not None:
        parts.append(vc.transform([text]))
    parts.append(sparse.csr_matrix(tabular))
    X_tfidf = sparse.hstack(parts).tocsr()

    # ── Predicción del ensemble (Exp1 + Exp2) ─────────────────────────────────
    clf_tfidf = _model['clf_tfidf']      # OvR LogReg entrenado en Exp 1
    w1        = _model['blend_w1']       # peso del modelo TF-IDF
    proba_tfidf = clf_tfidf.predict_proba(X_tfidf)[0]   # (24,)

    # Exp2: embeddings (sentence-transformers)
    encoder  = _model.get('encoder')
    clf_emb  = _model.get('clf_emb')
    w2       = 1.0 - w1

    if encoder is not None and clf_emb is not None:
        year_scaled_emb  = year_scaler.transform([[year]])
        emb              = encoder.encode([text], normalize_embeddings=True)
        X_emb            = np.hstack([emb, decade_vec, year_scaled_emb])
        proba_emb        = clf_emb.predict_proba(X_emb)[0]
        proba_final      = w1 * proba_tfidf + w2 * proba_emb
    else:
        proba_final      = proba_tfidf   # fallback: solo Exp1

    return {genre: round(float(p), 6)
            for genre, p in zip(GENRE_NAMES, proba_final)}


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Uso: python prediction.py "Título" "Plot..." 1994')
    else:
        result = predict_genres({
            'title': sys.argv[1],
            'plot':  sys.argv[2],
            'year':  int(sys.argv[3]),
        })
        for genre, prob in sorted(result.items(), key=lambda x: -x[1]):
            print(f'  {genre:<15} {prob:.4f}')
