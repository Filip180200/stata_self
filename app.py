from flask import Flask, render_template, request, session, make_response, url_for, redirect
import pandas as pd
import numpy as np
from scipy import stats
import uuid
import io
import hashlib
import os
import tempfile

# Sprawdzamy dostępność bibliotek (ważne dla stabilności)
try:
    from statsmodels.stats.diagnostic import lilliefors
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False

app = Flask(__name__)

# --- KONFIGURACJA SESJI ---
# Sekretny klucz jest niezbędny do działania sesji (zapamiętywania ID studenta).
# W Renderze najlepiej ustawić zmienną środowiskową SECRET_KEY, ale ten fallback zadziała.
app.secret_key = os.environ.get('SECRET_KEY', 'bardzo_tajny_klucz_deweloperski_123')

# Konfiguracja ciasteczek (ważne dla LTI w iframe w przyszłości)
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

# Hasło do panelu admina (zmiana ID w locie)
ADMIN_PASSWORD = "1234"

# -------------------------------------------------------------------------
# 1. LOGIKA GENEROWANIA DANYCH
# -------------------------------------------------------------------------
def generuj_dane_studenta(user_id, n=100):
    # Używamy ID studenta jako ziarna (seed) dla generatora liczb losowych.
    # Dzięki temu ten sam student ZAWSZE dostanie te same dane.
    seed_hex = hashlib.md5(str(user_id).encode('utf-8')).hexdigest()
    seed = int(seed_hex, 16) % (2**32)
    np.random.seed(seed)
    
    # Parametry do generowania macierzy kowariancji (Wielka Piątka)
    mean = [12, 12, 12, 13, 14] 
    cov_matrix = [
        [1.0, -0.6, 0.5, 0.3, 0.1],    
        [-0.6, 1.0, -0.4, -0.1, -0.2], 
        [0.5, -0.4, 1.0, 0.2, 0.2],    
        [0.3, -0.1, 0.2, 1.0, 0.3],    
        [0.1, -0.2, 0.2, 0.3, 1.0]     
    ]
    sd = 3.5
    cov_matrix = np.array(cov_matrix) * (sd ** 2)
    
    # Generowanie próbki
    data = np.random.multivariate_normal(mean, cov_matrix, size=n)
    data = np.rint(data).astype(int)
    data = np.clip(data, 5, 20) # Ograniczenie zakresu punktów
    
    df = pd.DataFrame(data, columns=['Ekstrawersja', 'Neurotyzm', 'Otwartosc', 'Ugodowosc', 'Sumiennosc'])
    return df

# -------------------------------------------------------------------------
# 2. KLUCZ ODPOWIEDZI (Samo-sprawdzanie)
# -------------------------------------------------------------------------
def ocen_sile(r):
    abs_r = abs(r)
    if abs_r < 0.1: return "brak"
    if abs_r < 0.3: return "słaby"
    if abs_r < 0.5: return "umiarkowany"
    return "silny"

def ocen_kierunek(r):
    if abs(r) < 0.1: return "związku"
    return "pozytywny" if r > 0 else "negatywny"

def format_r_z_gwiazdkami(r, p):
    stars = ""
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    return f"{r:.2f}{stars}".replace('.', ',')

PREFIX_MAP = {
    'Ekstrawersja': 'ekstra',
    'Neurotyzm': 'neuro',
    'Otwartosc': 'otwar',
    'Ugodowosc': 'ugoda',
    'Sumiennosc': 'sumien'
}

def oblicz_poprawne_statystyki(df):
    klucz = {}
    
    # A. Statystyki opisowe
    for col_name, prefix in PREFIX_MAP.items():
        series = df[col_name]
        klucz[f'{prefix}_m'] = series.mean()
        klucz[f'{prefix}_mdn'] = series.median()
        klucz[f'{prefix}_sd'] = series.std()
        klucz[f'{prefix}_sk'] = series.skew()
        klucz[f'{prefix}_kurt'] = series.kurt()
        klucz[f'{prefix}_min'] = series.min()
        klucz[f'{prefix}_max'] = series.max()
        
        # Test normalności
        if HAS_STATSMODELS:
            try:
                d_stat, p_val = lilliefors(series, dist='norm')
            except:
                d_stat, p_val = 0, 1.0
        else:
            z_score = (series - series.mean()) / (series.std() or 1)
            d_stat, p_val = stats.kstest(z_score, 'norm')
            
        klucz[f'{prefix}_d'] = d_stat
        klucz[f'{prefix}_p'] = p_val

    # B. Decyzja o metodzie (parametryczna vs nieparametryczna)
    # Przyjmujemy założenie: jeśli neurotyzm nie ma rozkładu normalnego -> Spearman
    if klucz['neuro_p'] < 0.05:
        klucz['rozklad_typ'] = 'niespełnienie'
        klucz['korelacja_typ'] = 'rho-Spearmana'
    else:
        klucz['rozklad_typ'] = 'spełnienie'
        klucz['korelacja_typ'] = 'r-Pearsona'

    # C. Obliczanie korelacji (Spearman jako bezpieczny default dla klucza w tym zadaniu)
    vars_list = ['Ekstrawersja', 'Neurotyzm', 'Otwartosc', 'Ugodowosc', 'Sumiennosc']
    rho, p_matrix = stats.spearmanr(df[vars_list])
    idx = {name: i for i, name in enumerate(vars_list)}
    
    def get_rp(v1, v2):
        i, j = idx[v1], idx[v2]
        return rho[i, j], p_matrix[i, j]

    # Mapowanie korelacji do pól formularza
    pairs = [
        ('Neurotyzm', 'Ekstrawersja', 'corr_neuro_ekstra'),
        ('Otwartosc', 'Ekstrawersja', 'corr_otwar_ekstra'),
        ('Otwartosc', 'Neurotyzm', 'corr_otwar_neuro'),
        ('Ugodowosc', 'Ekstrawersja', 'corr_ugoda_ekstra'),
        ('Ugodowosc', 'Neurotyzm', 'corr_ugoda_neuro'),
        ('Ugodowosc', 'Otwartosc', 'corr_ugoda_otwar'),
        ('Sumiennosc', 'Ekstrawersja', 'corr_sumien_ekstra'),
        ('Sumiennosc', 'Neurotyzm', 'corr_sumien_neuro'),
        ('Sumiennosc', 'Otwartosc', 'corr_sumien_otwar'),
        ('Sumiennosc', 'Ugodowosc', 'corr_sumien_ugoda'),
    ]

    for v1, v2, key in pairs:
        r, p = get_rp(v1, v2)
        klucz[key] = format_r_z_gwiazdkami(r, p)

    # D. Weryfikacja hipotez (H1-H4)
    # H1: Ekstrawersja a Neurotyzm
    r1, p1 = get_rp('Neurotyzm', 'Ekstrawersja')
    klucz['h1_sila'] = ocen_sile(r1)
    klucz['h1_kierunek'] = ocen_kierunek(r1)
    klucz['h1_decyzja'] = 'potwierdza' if (p1 < 0.05 and r1 < 0) else 'nie potwierdza'

    # H2: Ekstrawersja a Otwartość
    r2, p2 = get_rp('Otwartosc', 'Ekstrawersja')
    klucz['h2_sila'] = ocen_sile(r2)
    klucz['h2_kierunek'] = ocen_kierunek(r2)
    klucz['h2_decyzja'] = 'potwierdza' if (p2 < 0.05 and r2 > 0) else 'nie potwierdza'

    # H3: Neurotyzm a Otwartość
    r3, p3 = get_rp('Otwartosc', 'Neurotyzm')
    klucz['h3_sila'] = ocen_sile(r3)
    klucz['h3_kierunek'] = ocen_kierunek(r3)
    klucz['h3_decyzja'] = 'potwierdza' if (p3 < 0.05 and r3 < 0) else 'nie potwierdza' # Tu zależy od treści H3

    # H4: Ekstrawersja a Ugodowość
    r4, p4 = get_rp('Ugodowosc', 'Ekstrawersja')
    klucz['h4_sila'] = ocen_sile(r4)
    klucz['h4_kierunek'] = ocen_kierunek(r4)
    # Zakładamy że hipoteza przewidywała związek pozytywny
    if abs(r4) < 0.1 or p4 >= 0.05:
         klucz['h4_decyzja'] = 'nie potwierdza'
    else:
         klucz['h4_decyzja'] = 'potwierdza' if r4 > 0 else 'nie potwierdza'

    return klucz

# -------------------------------------------------------------------------
# 3. ROUTING I OBSŁUGA ID STUDENTA
# -------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    # 1. Obsługa ręcznej zmiany ID (Panel Admina lub formularz debugowania)
    if request.method == 'POST':
        manual_id = request.form.get('manual_id')
        if manual_id:
            session['user_id'] = manual_id.strip()
            # Przeładowujemy stronę (POST -> GET) żeby zapobiec ponownemu wysłaniu formularza
            return redirect(url_for('index', **request.args))

    # 2. Sprawdzenie, czy ID jest w URL (np. ?user_id=JanKowalski)
    # To ma priorytet nad sesją.
    user_id_from_url = request.args.get('user_id')
    if user_id_from_url:
        session['user_id'] = user_id_from_url
    
    # 3. AUTOMATYCZNE GENEROWANIE ID
    # Jeśli nie ma ID w sesji (czyli student wszedł pierwszy raz), generujemy mu losowe ID.
    if 'user_id' not in session:
        # Generujemy krótki, losowy ciąg znaków (np. a1b2c3d4)
        random_id = str(uuid.uuid4())[:8]
        session['user_id'] = random_id

    # Pobieramy finalne ID z sesji
    user_id = session['user_id']
    
    # Generujemy dane i klucz odpowiedzi
    df = generuj_dane_studenta(user_id)
    answers = oblicz_poprawne_statystyki(df)
    
    return render_template('index.html', user_id=user_id, answers=answers)

@app.route('/pobierz_csv')
def pobierz_csv():
    if 'user_id' not in session: return redirect(url_for('index'))
    
    user_id = session['user_id']
    df = generuj_dane_studenta(user_id)
    
    output = io.BytesIO()
    df.to_csv(output, index=False, sep=';', encoding='utf-8')
    output.seek(0)
    
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=dane_{user_id}.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/pobierz_sav')
def pobierz_sav():
    if 'user_id' not in session: return redirect(url_for('index'))
    
    if not HAS_PYREADSTAT:
        return "Błąd serwera: Biblioteka pyreadstat nie jest zainstalowana.", 500

    user_id = session['user_id']
    df = generuj_dane_studenta(user_id)
    
    # Trik z plikiem tymczasowym, bo pyreadstat nie umie pisać bezpośrednio do strumienia bajtów
    fd, path = tempfile.mkstemp(suffix='.sav')
    try:
        os.close(fd) 
        import pyreadstat
        pyreadstat.write_sav(df, path)
        
        with open(path, 'rb') as f:
            data = f.read()
            
        output = io.BytesIO(data)
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = f"attachment; filename=dane_{user_id}.sav"
        response.headers["Content-type"] = "application/x-spss-sav"
        return response
    except Exception as e:
        return f"Wystąpił błąd SAV: {e}", 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# Trasa pomocnicza, jeśli formularz wysyła POST na /sprawdz (choć teraz robimy to w JS)
@app.route('/sprawdz', methods=['GET', 'POST'])
def sprawdz():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)