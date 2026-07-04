# Ce fichier est la "Base" des features.
# Son role : appeler toutes les fonctions d'extraction (URL + HTML)
# et retourner un dictionnaire de 30 features dans l'ORDRE EXACT
# attendu par le modele ML entraine.
# Convention : 1 = Phishing, -1 = Legitime, 0 = Suspect

import sys
import os

# Ajoute le dossier parent au chemin de recherche Python
# pour pouvoir importer url_features et html_features
# depuis n'importe ou (pas seulement depuis le dossier features/)
sys.path.append(os.path.dirname(__file__))

# Import de toutes les fonctions d'extraction depuis url_features.py
# Chaque fonction prend une URL en entree et retourne -1, 0 ou 1
from url_features import (
    using_ip,                 # Detecte si l'URL contient une IP brute
    long_url,                 # Mesure la longueur de l'URL
    short_url,                # Detecte les raccourcisseurs (bit.ly, tinyurl...)
    symbol_at,                # Detecte le symbole @ dans l'URL
    redirecting_double_slash, # Detecte une redirection cachee via //
    prefix_suffix,            # Detecte un tiret dans le nom de domaine
    sub_domains,              # Compte le nombre de sous-domaines
    https_feature,            # Verifie la presence de HTTPS
    non_std_port,             # Detecte un port inhabituel dans l'URL
    https_domain_url,         # Detecte "https" dans le nom de domaine lui-meme
    info_email,               # Detecte la presence de mailto: dans l'URL
    favicon,                  # Detecte si le favicon vient d'un domaine externe
    abnormal_url,             # Verifie si le domaine est resolvable en DNS
    dns_recording,            # Verifie si le domaine a un enregistrement DNS
    domain_reg_len,           # Verifie la duree d'enregistrement du domaine (WHOIS)
    age_of_domain             # Verifie l'age du domaine depuis sa creation (WHOIS)
)

# Import depuis html_features.py
# telecharger_page : telecharge le HTML brut d'une page
# extract_html_features : calcule les 14 features HTML en une seule fois
from html_features import telecharger_page, extract_html_features


def extraire_toutes_les_features(url):
    """
    Fonction principale du pipeline.
    Prend une URL en entree et retourne un dictionnaire de 30 features
    dans l'ordre exact attendu par le modele (verifie via feature_names.pkl).

    Pourquoi cet ordre est important ?
    Le modele a ete entraine avec les colonnes dans un ordre precis.
    Si on les passe dans le mauvais ordre, le modele interprete
    une feature a la place d'une autre -> predictions fausses.

    Convention des valeurs :
      1  = signe de Phishing
     -1  = signe de Legitimite
      0  = Suspect (valeur intermediaire ou donnees non disponibles)
    """

    # ── TELECHARGEMENT DU HTML ──────────────────────────────────
    # On telecharge la page UNE SEULE FOIS ici pour la feature Favicon
    # (qui a besoin du HTML brut pour chercher la balise <link rel="icon">)
    html = telecharger_page(url)

    # extract_html_features telecharge la page une deuxieme fois en interne.
    # Limite connue : double telechargement -> legere latence supplementaire.
    # Amelioration possible : passer html directement a extract_html_features.
    html_feats = extract_html_features(url)

    # ── CONSTRUCTION DU VECTEUR DE 30 FEATURES ─────────────────
    # L'ordre des cles ici correspond EXACTEMENT a l'ordre des colonnes
    # du dataset UCI original (verifie via joblib.load('feature_names.pkl'))
    features = {

        # === FEATURES BASEES SUR L'URL (calcul instantane) ===

        # 0 - IP brute dans l'URL au lieu d'un nom de domaine
        'UsingIP':            using_ip(url),

        # 1 - URL tres longue pour cacher la partie suspecte
        'LongURL':            long_url(url),

        # 2 - Raccourcisseur d'URL (cache la vraie destination)
        'ShortURL':           short_url(url),

        # 3 - @ dans l'URL (force le navigateur a ignorer tout ce qui precede)
        'Symbol@':            symbol_at(url),

        # 4 - // apres la position 7 (redirection cachee)
        'Redirecting//':      redirecting_double_slash(url),

        # 5 - Tiret dans le nom de domaine (ex: paypal-secure.com)
        'PrefixSuffix-':      prefix_suffix(url),

        # 6 - Nombre de sous-domaines (ex: secure.login.verify.malicious.com)
        'SubDomains':         sub_domains(url),

        # 7 - Presence du protocole HTTPS
        'HTTPS':              https_feature(url),

        # 8 - Duree d'enregistrement du domaine (via WHOIS, lent ~1-3s)
        'DomainRegLen':       domain_reg_len(url),

        # 9 - Favicon charge depuis un domaine externe (necessite le HTML)
        'Favicon':            favicon(url, html),

        # 10 - Port non standard explicitement precise dans l'URL
        'NonStdPort':         non_std_port(url),

        # 11 - Le mot "https" utilise dans le nom de domaine lui-meme
        'HTTPSDomainURL':     https_domain_url(url),

        # === FEATURES BASEES SUR LE HTML (necessitent de telecharger la page) ===

        # 12 - % de ressources (images, scripts) chargees depuis un domaine externe
        'RequestURL':         html_feats['RequestURL'],

        # 13 - % de liens <a> pointant vers un domaine externe
        'AnchorURL':          html_feats['AnchorURL'],

        # 14 - % de liens dans les balises <script>, <link>, <meta>
        'LinksInScriptTags':  html_feats['LinksInScriptTags'],

        # 15 - Destination du formulaire de connexion (SFH = Server Form Handler)
        'ServerFormHandler':  html_feats['ServerFormHandler'],

        # 16 - Presence de mailto: dans l'URL (envoi d'infos par email)
        'InfoEmail':          info_email(url),

        # 17 - Coherence entre le nom d'hote et le DNS (approxime)
        'AbnormalURL':        abnormal_url(url),

        # 18 - Nombre de redirections (meta refresh)
        'WebsiteForwarding':  html_feats['WebsiteForwarding'],

        # 19 - Modification de la barre de statut via onMouseOver
        'StatusBarCust':      html_feats['StatusBarCust'],

        # 20 - Desactivation du clic droit via JavaScript
        'DisableRightClick':  html_feats['DisableRightClick'],

        # 21 - Popup avec champ de texte pour voler des identifiants
        'UsingPopupWindow':   html_feats['UsingPopupWindow'],

        # 22 - Presence d'iframes (souvent invisibles chez les phishers)
        'IframeRedirection':  html_feats['IframeRedirection'],

        # === FEATURES BASEES SUR LE DOMAINE (WHOIS + DNS) ===

        # 23 - Age du domaine depuis sa creation (via WHOIS, lent ~1-3s)
        'AgeofDomain':        age_of_domain(url),

        # 24 - Existence d'un enregistrement DNS pour ce domaine
        'DNSRecording':       dns_recording(url),

        # === FEATURES NON DISPONIBLES (APIs fermees) ===
        # Ces features existaient dans le dataset original mais leurs sources
        # de donnees ne sont plus accessibles gratuitement aujourd'hui.
        # On utilise des valeurs par defaut documentees dans le rapport.

        # 25 - Popularite du site (Alexa Rank - service ferme en 2022)
        'WebsiteTraffic':     html_feats['WebsiteTraffic'],    # -> 0 (suspect) par defaut

        # 26 - PageRank Google (API fermee depuis 2016)
        'PageRank':           html_feats['PageRank'],           # -> -1 (neutre) par defaut

        # 27 - Site indexe par Google (scraping Google bloque)
        'GoogleIndex':        html_feats['GoogleIndex'],        # -> -1 (neutre) par defaut

        # 28 - Nombre de liens externes pointant vers cette page (Ahrefs/Moz = payant)
        'LinksPointingToPage': html_feats['LinksPointingToPage'], # -> 0 (suspect) par defaut

        # 29 - Site dans les listes noires PhishTank/StopBadware (quota API depasse)
        'StatsReport':        html_feats['StatsReport'],        # -> -1 (neutre) par defaut
    }

    return features  # dictionnaire de 30 cles -> valeurs -1/0/1


# ── TEST RAPIDE ─────────────────────────────────────────────────
# Ce bloc ne s'execute que si on lance ce fichier directement
# (pas quand il est importe par app.py ou un autre script)
if __name__ == '__main__':
    url = 'https://www.google.com'
    print(f"Test complet sur {url}\n")
    feats = extraire_toutes_les_features(url)
    # Affiche chaque feature avec son index (pour verifier l'ordre)
    for i, (k, v) in enumerate(feats.items()):
        print(f"{i}: {k} = {v}")