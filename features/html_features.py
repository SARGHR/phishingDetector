# Ce fichier extrait les 14 features basees sur le contenu HTML de la page cible.
# Ces features necessitent de telecharger la page web (via requests + BeautifulSoup).
# Si la page est injoignable (domaine mort, timeout...), toutes les features
# retournent 0 (Suspect) car un domaine injoignable est souvent un phishing ephemere.
#
# Convention : 1 = Phishing, -1 = Legitime, 0 = Suspect
# Source des regles : Mohammad, Thabtah & McCluskey (2014) - UCI Phishing Dataset

import requests
from bs4 import BeautifulSoup
import tldextract


def telecharger_page(url, timeout=5):
    # Telecharge le HTML brut d'une page web.
    # timeout=5 : on attend max 5 secondes avant d'abandonner.
    # User-Agent : on se fait passer pour un navigateur normal
    # pour eviter d'etre bloque par les sites.
    # Retourne le HTML sous forme de chaine de caracteres,
    # ou None si le telechargement a echoue (timeout, erreur reseau, 404...).
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; PhishingDetector/1.0)'}
        response = requests.get(url, timeout=timeout, headers=headers)
        return response.text  # le HTML brut de la page
    except Exception:
        return None  # echec -> on retourne None, gere par chaque fonction


def request_url(url, html):
    # Feature RequestURL : mesure le % de ressources (images, scripts, css)
    # chargees depuis un domaine DIFFERENT de celui de la page principale.
    # Un site legitime charge ses ressources depuis son propre domaine.
    # Un site de phishing copie souvent des ressources depuis des sites externes.
    #
    # Regle officielle UCI :
    #   ratio < 22%  -> Legitime (-1)
    #   22% <= ratio <= 61% -> Suspect (0)
    #   ratio > 61%  -> Phishing (1)
    if html is None:
        return -1  # page injoignable -> valeur neutre par defaut ici
    ext_site = tldextract.extract(url)  # decoupe l'URL en sous-domaine/domaine/extension
    soup = BeautifulSoup(html, 'html.parser')  # parse le HTML pour naviguer dedans

    total = 0    # nombre total de ressources trouvees
    externes = 0 # nombre de ressources venant d'un autre domaine

    # On cherche toutes les balises img (images), script (JS), link (CSS)
    for tag in soup.find_all(['img', 'script', 'link']):
        src = tag.get('src') or tag.get('href')  # l'attribut qui contient l'URL de la ressource
        if src and src.startswith('http'):        # on ignore les chemins relatifs (/images/logo.png)
            total += 1
            ext_src = tldextract.extract(src)
            if ext_src.domain != ext_site.domain: # domaine different = ressource externe
                externes += 1

    if total == 0:
        return -1  # pas de ressources trouvees -> pas de risque detecte
    ratio = externes / total
    if ratio < 0.22:
        return -1
    elif ratio <= 0.61:
        return 0
    else:
        return 1


def anchor_url(url, html):
    # Feature AnchorURL : mesure le % de liens <a href="...">
    # qui pointent vers un domaine DIFFERENT de la page principale.
    # Les phishers copient des pages legitimes et laissent des liens
    # qui pointent vers les vrais sites (donc vers des domaines externes).
    #
    # Regle officielle UCI :
    #   ratio < 31%  -> Legitime (-1)
    #   31% <= ratio <= 67% -> Suspect (0)
    #   ratio > 67%  -> Phishing (1)
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    total = 0
    externes = 0

    for tag in soup.find_all('a'):         # toutes les balises <a>
        href = tag.get('href')             # l'URL du lien
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
    # Feature LinksInScriptTags : mesure le % de liens externes
    # dans les balises <script>, <link> et <meta>.
    # Meme logique que RequestURL mais ciblee sur ces 3 balises specifiques
    # qui sont souvent utilisees pour charger du code malveillant externe.
    #
    # Regle officielle UCI :
    #   ratio < 17%  -> Legitime (-1)
    #   17% <= ratio <= 81% -> Suspect (0)
    #   ratio > 81%  -> Phishing (1)
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    total = 0
    externes = 0

    for tag in soup.find_all(['script', 'link', 'meta']):
        # chaque balise peut avoir son URL dans src, href ou content
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
    # Feature ServerFormHandler (SFH) : verifie ou le formulaire de connexion
    # envoie les donnees saisies par l'utilisateur.
    # Un phisheur redirige ces donnees vers son propre serveur.
    #
    # Regle officielle UCI :
    #   action vide ou "about:blank" -> Phishing (1)
    #   action pointe vers un domaine different -> Suspect (0)
    #   action pointe vers le meme domaine -> Legitime (-1)
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    soup = BeautifulSoup(html, 'html.parser')

    forms = soup.find_all('form')  # toutes les balises <form>
    if not forms:
        return -1  # pas de formulaire = pas de risque pour cette feature

    for form in forms:
        action = form.get('action', '')  # l'attribut action du formulaire
        if action == '' or action.lower() == 'about:blank':
            # formulaire vide ou pointe vers "about:blank" = tres suspect
            return 1
        if action.startswith('http'):
            ext_action = tldextract.extract(action)
            if ext_action.domain != ext_site.domain:
                # le formulaire envoie les donnees vers un autre domaine
                return 0
    return -1  # tout est sur le meme domaine -> legitime


def website_forwarding(html):
    # Feature WebsiteForwarding : compte le nombre de redirections automatiques
    # via la balise <meta http-equiv="refresh">.
    # Les sites de phishing redirigent souvent plusieurs fois pour brouiller les pistes.
    #
    # Regle officielle UCI :
    #   0 ou 1 redirection -> Legitime (-1)
    #   2 ou 3 redirections -> Suspect (0)
    #   4 redirections ou plus -> Phishing (1)
    if html is None:
        return -1
    soup = BeautifulSoup(html, 'html.parser')
    # cherche toutes les balises <meta http-equiv="refresh">
    meta_refresh = soup.find_all('meta', attrs={'http-equiv': lambda x: x and x.lower() == 'refresh'})
    nb = len(meta_refresh)
    if nb <= 1:
        return -1
    elif nb <= 3:
        return 0
    else:
        return 1


def iframe_redirection(html):
    # Feature IframeRedirection : detecte la presence de balises <iframe>.
    # Les phisheurs utilisent des iframes invisibles pour superposer
    # une fausse page par-dessus une vraie page.
    #
    # Regle officielle UCI :
    #   iframe present -> Phishing (1)
    #   aucun iframe -> Legitime (-1)
    if html is None:
        return -1
    soup = BeautifulSoup(html, 'html.parser')
    return 1 if soup.find_all('iframe') else -1


def disable_right_click(html):
    # Feature DisableRightClick : detecte si le site desactive le clic droit
    # via JavaScript pour empecher l'utilisateur d'inspecter le code source.
    # Technique classique des phisheurs pour cacher leur code.
    #
    # Regle officielle UCI :
    #   clic droit desactive -> Phishing (1)
    #   sinon -> Legitime (-1)
    #
    # Limite connue : certains sites legitimes complexes (ex: Google)
    # contiennent aussi ces patterns pour d'autres raisons -> faux positifs possibles.
    if html is None:
        return -1
    html_lower = html.lower()
    # on cherche des patterns JavaScript typiques de la desactivation du clic droit
    patterns = ['event.button==2', 'oncontextmenu', 'contextmenu', 'preventdefault()']
    return 1 if any(p in html_lower for p in patterns) else -1


def status_bar_cust(html):
    # Feature StatusBarCust : detecte si le site modifie la barre de statut
    # du navigateur via JavaScript (onMouseOver).
    # Technique utilisee pour afficher une fausse URL dans la barre en bas
    # du navigateur quand l'utilisateur survole un lien.
    #
    # Regle officielle UCI :
    #   onMouseOver modifie la barre -> Phishing (1)
    #   sinon -> Legitime (-1)
    if html is None:
        return -1
    html_lower = html.lower()
    return 1 if 'window.status' in html_lower or 'onmouseover' in html_lower else -1


def using_popup_window(html):
    # Feature UsingPopupWindow : detecte si le site ouvre une popup
    # contenant un champ de texte pour voler des identifiants.
    # Un site legitime ne demande jamais d'informations personnelles
    # via une popup JavaScript.
    #
    # Regle officielle UCI :
    #   popup avec champ texte -> Phishing (1)
    #   sinon -> Legitime (-1)
    if html is None:
        return -1
    html_lower = html.lower()
    # window.open( = ouverture d'une popup + 'text' = champ de saisie present
    return 1 if 'window.open(' in html_lower and 'text' in html_lower else -1


def extract_html_features(url):
    # Fonction principale de ce fichier.
    # Telecharge la page UNE SEULE FOIS et calcule les 14 features HTML.
    #
    # Cas special : page injoignable (html = None)
    # Un domaine completement injoignable est souvent un site de phishing
    # ephemere qui a ete supprime apres la campagne.
    # -> On retourne 0 (Suspect) pour toutes les features
    #    plutot que -1 (Legitime) par defaut.
    html = telecharger_page(url)

    if html is None:
        # page injoignable : on ne peut rien verifier -> tout est suspect
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
            # Features non disponibles aussi mises a 0 (suspect) si page injoignable
            'PageRank': 0,
            'WebsiteTraffic': 0,
            'GoogleIndex': 0,
            'LinksPointingToPage': 0,
            'StatsReport': 0,
        }

    # page accessible : on calcule chaque feature normalement
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

        # Features dont les sources de donnees ne sont plus disponibles gratuitement.
        # Valeurs par defaut choisies selon les statistiques du papier original :

        # PageRank Google (API fermee depuis 2016)
        # -> -1 par defaut (on suppose legitime, valeur optimiste)
        'PageRank': -1,

        # Alexa Rank (service ferme en 2022)
        # -> 0 par defaut (on ne sait pas = suspect)
        'WebsiteTraffic': 0,

        # Indexation Google (scraping bloque)
        # -> -1 par defaut (on suppose indexe = legitime)
        'GoogleIndex': -1,

        # Backlinks (Ahrefs/Moz = payant)
        # -> 0 par defaut (on ne sait pas = suspect)
        'LinksPointingToPage': 0,

        # Listes noires PhishTank/StopBadware (quota API)
        # -> -1 par defaut (on suppose non liste = legitime)
        'StatsReport': -1,
    }


# ── TEST RAPIDE ─────────────────────────────────────────────────
# Ce bloc ne s'execute que si on lance ce fichier directement
if __name__ == '__main__':
    url = 'https://www.google.com'
    print(f"Test sur {url}")
    resultats = extract_html_features(url)
    for k, v in resultats.items():
        print(f"  {k}: {v}")