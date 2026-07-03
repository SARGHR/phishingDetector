import requests
from bs4 import BeautifulSoup
import tldextract


def telecharger_page(url, timeout=5):
    """Télécharge le HTML d'une page une seule fois.
    Retourne le HTML (str) si succès, None si échec."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; PhishingDetector/1.0)'}
        response = requests.get(url, timeout=timeout, headers=headers)
        return response.text
    except Exception:
        return None


def request_url(url, html):
    """RequestURL : % de ressources chargées depuis un domaine externe
    Règle officielle : <22% → Legitimate(-1), 22-61% → Suspicious(0), >61% → Phishing(1)"""
    if html is None:
        return -1  # valeur neutre légitime si on n'a pas pu télécharger
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    total = 0
    externes = 0
    for tag in soup.find_all(['img', 'script', 'link']):
        src = tag.get('src') or tag.get('href')
        if src and src.startswith('http'):
            total += 1
            ext_src = tldextract.extract(src)
            if ext_src.domain != ext_site.domain:
                externes += 1

    if total == 0:
        return -1
    ratio = externes / total
    if ratio < 0.22:
        return -1
    elif ratio <= 0.61:
        return 0
    else:
        return 1


def anchor_url(url, html):
    """AnchorURL : % de liens <a> qui pointent vers un domaine externe
    Règle officielle : <31% → Legitimate(-1), 31-67% → Suspicious(0), >67% → Phishing(1)"""
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    total = 0
    externes = 0
    for tag in soup.find_all('a'):
        href = tag.get('href')
        if href and href.startswith('http'):
            total += 1
            ext_href = tldextract.extract(href)
            if ext_href.domain != ext_site.domain:
                externes += 1

    if total == 0:
        return -1
    ratio = externes / total
    if ratio < 0.31:
        return -1
    elif ratio <= 0.67:
        return 0
    else:
        return 1


def links_in_script_tags(url, html):
    """LinksInScriptTags : liens externes dans <meta>/<script>/<link>
    Règle officielle : <17% → Legitimate(-1), 17-81% → Suspicious(0), >81% → Phishing(1)"""
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    total = 0
    externes = 0
    for tag in soup.find_all(['script', 'link', 'meta']):
        src = tag.get('src') or tag.get('href') or tag.get('content')
        if src and src.startswith('http'):
            total += 1
            ext_src = tldextract.extract(src)
            if ext_src.domain != ext_site.domain:
                externes += 1

    if total == 0:
        return -1
    ratio = externes / total
    if ratio < 0.17:
        return -1
    elif ratio <= 0.81:
        return 0
    else:
        return 1


def server_form_handler(url, html):
    """ServerFormHandler : où le formulaire envoie les données
    Règle officielle : vide/about:blank → Phishing(1), domaine différent → Suspicious(0), sinon → Legitimate(-1)"""
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    forms = soup.find_all('form')
    if not forms:
        return -1  # pas de formulaire = pas de risque ici

    for form in forms:
        action = form.get('action', '')
        if action == '' or action.lower() == 'about:blank':
            return 1
        if action.startswith('http'):
            ext_action = tldextract.extract(action)
            if ext_action.domain != ext_site.domain:
                return 0
    return -1


def website_forwarding(html):
    """WebsiteForwarding : nombre de redirections (meta refresh)
    Règle officielle : ≤1 → Legitimate(-1), 2-3 → Suspicious(0), ≥4 → Phishing(1)"""
    if html is None:
        return -1
    soup = BeautifulSoup(html, 'html.parser')
    meta_refresh = soup.find_all('meta', attrs={'http-equiv': lambda x: x and x.lower() == 'refresh'})
    nb = len(meta_refresh)
    if nb <= 1:
        return -1
    elif nb <= 3:
        return 0
    else:
        return 1


def iframe_redirection(html):
    """IframeRedirection : présence d'iframes (même visibles)
    Règle officielle : iframe utilisé → Phishing(1), sinon → Legitimate(-1)"""
    if html is None:
        return -1
    soup = BeautifulSoup(html, 'html.parser')
    return 1 if soup.find_all('iframe') else -1


def disable_right_click(html):
    """DisableRightClick : le site désactive le clic droit
    Règle officielle : event.button==2 trouvé → Phishing(1), sinon → Legitimate(-1)"""
    if html is None:
        return -1
    html_lower = html.lower()
    patterns = ['event.button==2', 'oncontextmenu', 'contextmenu', 'preventdefault()']
    return 1 if any(p in html_lower for p in patterns) else -1


def status_bar_cust(html):
    """StatusBarCust : la barre de statut du navigateur est modifiée
    Règle officielle : onMouseOver change la barre → Phishing(1), sinon → Legitimate(-1)"""
    if html is None:
        return -1
    html_lower = html.lower()
    return 1 if 'window.status' in html_lower or 'onmouseover' in html_lower else -1


def using_popup_window(html):
    """UsingPopupWindow : ouverture de popups avec champs de texte
    Règle officielle : popup avec champ texte → Phishing(1), sinon → Legitimate(-1)"""
    if html is None:
        return -1
    html_lower = html.lower()
    return 1 if 'window.open(' in html_lower and 'text' in html_lower else -1


def extract_html_features(url):
    """Fonction principale : télécharge la page UNE FOIS
    et calcule toutes les features HTML d'un coup.
    Si la page est injoignable, c'est un signal suspect en soi (souvent domaine
    de phishing éphémère) → toutes les features HTML basculent en Suspicious(0)
    plutôt que Legitimate(-1) par défaut."""
    html = telecharger_page(url)

    if html is None:
        # Page injoignable : on ne peut RIEN vérifier → suspect partout
        return {
            'RequestURL': 0,
            'AnchorURL': 0,
            'LinksInScriptTags': 0,
            'ServerFormHandler': 0,
            'WebsiteForwarding': 0,
            'IframeRedirection': 0,
            'DisableRightClick': 0,
            'StatusBarCust': 0,
            'UsingPopupWindow': 0,
            'PageRank': 0,
            'WebsiteTraffic': 0,
            'GoogleIndex': 0,
            'LinksPointingToPage': 0,
            'StatsReport': 0,
        }

    return {
        'RequestURL': request_url(url, html),
        'AnchorURL': anchor_url(url, html),
        'LinksInScriptTags': links_in_script_tags(url, html),
        'ServerFormHandler': server_form_handler(url, html),
        'WebsiteForwarding': website_forwarding(html),
        'IframeRedirection': iframe_redirection(html),
        'DisableRightClick': disable_right_click(html),
        'StatusBarCust': status_bar_cust(html),
        'UsingPopupWindow': using_popup_window(html),
        'PageRank': -1,
        'WebsiteTraffic': 0,
        'GoogleIndex': -1,
        'LinksPointingToPage': 0,
        'StatsReport': -1,
    }

if __name__ == '__main__':
    url = 'https://www.google.com'
    print(f"Test sur {url}")
    resultats = extract_html_features(url)
    for k, v in resultats.items():
        print(f"  {k}: {v}")