# Ce fichier extrait les 16 features basees sur la structure de l'URL.
# Toutes ces features sont calculees instantanement (pas de telechargement de page).
# Certaines font des requetes reseau (WHOIS, DNS) qui peuvent prendre 1-3 secondes.
#
# Convention : 1 = Phishing, -1 = Legitime, 0 = Suspect
# Source des regles : Mohammad, Thabtah & McCluskey (2014) - UCI Phishing Dataset

import re       # pour les expressions regulieres (recherche de patterns dans les URLs)
import socket   # pour les requetes DNS (verifier si un domaine existe)
import tldextract  # pour decouper une URL en sous-domaine / domaine / extension


def using_ip(url):
    # Feature UsingIP : detecte si l'URL utilise une adresse IP brute
    # a la place d'un nom de domaine.
    # Ex: http://125.98.3.123/fake.html -> suspect
    #     http://0x58.0xCC.0xCA.0x62/paypal -> aussi suspect (IP en hexadecimal)
    # Un vrai site utilise toujours un nom de domaine, jamais une IP brute.
    #
    # Regle officielle UCI :
    #   IP presente -> Phishing (1)
    #   sinon -> Legitime (-1)
    ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|(0x[0-9a-fA-F]+)'
    # \d{1,3} = 1 a 3 chiffres, \. = un point, repete 4 fois pour matcher xxx.xxx.xxx.xxx
    # | = OU, 0x[0-9a-fA-F]+ = IP en notation hexadecimale
    return 1 if re.search(ip_pattern, url) else -1


def long_url(url):
    # Feature LongURL : mesure la longueur totale de l'URL.
    # Les phisheurs allongent l'URL pour cacher la partie suspecte
    # dans la barre d'adresse du navigateur.
    # Ex: http://federmacedoadv.com.br/3f/aze/ab51e2...@phishing.website.html
    #
    # Regle officielle UCI (basee sur les statistiques du dataset) :
    #   longueur < 54  -> Legitime (-1)
    #   54 <= longueur <= 75 -> Suspect (0)
    #   longueur > 75  -> Phishing (1)
    length = len(url)
    if length < 54:
        return -1
    elif length <= 75:
        return 0
    else:
        return 1


def short_url(url):
    # Feature ShortURL : detecte si l'URL utilise un service de raccourcissement.
    # Les raccourcisseurs (bit.ly, tinyurl...) cachent la vraie destination.
    # La victime voit "bit.ly/3xK9p2" sans savoir ou elle va atterrir.
    #
    # Regle officielle UCI :
    #   raccourcisseur detecte -> Phishing (1)
    #   sinon -> Legitime (-1)
    shorteners = r'bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly'
    # \. = un point litteral (sans \ le . voudrait dire "n'importe quel caractere")
    # | = OU entre chaque raccourcisseur
    return 1 if re.search(shorteners, url.lower()) else -1


def symbol_at(url):
    # Feature Symbol@ : detecte la presence du symbole @ dans l'URL.
    # Le navigateur ignore tout ce qui precede le @ dans une URL.
    # Ex: http://google.com@malicious.com -> le navigateur va sur malicious.com
    #     La victime voit "google.com" mais va sur le site malveillant.
    #
    # Regle officielle UCI :
    #   @ present -> Phishing (1)
    #   sinon -> Legitime (-1)
    return 1 if '@' in url else -1


def redirecting_double_slash(url):
    # Feature Redirecting// : detecte une redirection cachee via //.
    # Si // apparait apres la position 7 (apres "http://"),
    # cela signifie qu'il y a une deuxieme URL cachee dans le chemin.
    # Ex: http://www.legitime.com//http://www.phishing.com
    #     -> le navigateur va sur phishing.com
    #
    # Regle officielle UCI :
    #   position du DERNIER // > 7 -> Phishing (1)
    #   sinon -> Legitime (-1)
    #
    # rfind('//') retourne la position du DERNIER // dans l'URL.
    # Pour "http://site.com" le // est en position 5 -> normal -> -1
    # Pour "http://site.com//http://evil.com" le dernier // est bien apres 7 -> 1
    return 1 if url.rfind('//') > 6 else -1


def prefix_suffix(url):
    # Feature PrefixSuffix- : detecte un tiret dans le NOM DE DOMAINE.
    # Les phisheurs ajoutent des tirets pour imiter des marques connues.
    # Ex: paypal-secure.com, confirm-paypal.com, apple-id-login.com
    # Les vrais domaines de grandes marques n'ont presque jamais de tiret.
    #
    # Regle officielle UCI :
    #   tiret dans le domaine -> Phishing (1)
    #   sinon -> Legitime (-1)
    ext = tldextract.extract(url)  # decoupe l'URL : ext.domain = "paypal-secure" par ex
    return 1 if '-' in ext.domain else -1


def sub_domains(url):
    # Feature SubDomains : compte le nombre de sous-domaines dans l'URL.
    # Les phisheurs empilent des sous-domaines pour faire croire que
    # l'URL appartient a une marque connue.
    # Ex: secure.login.verify.paypal.malicious.com
    #     -> le vrai domaine est malicious.com, pas paypal
    #
    # Methode : on retire "www." s'il est present, puis on compte
    # les parties restantes du sous-domaine.
    #
    # Regle officielle UCI :
    #   0 ou 1 sous-domaine -> Legitime (-1)
    #   2 sous-domaines -> Suspect (0)
    #   3 sous-domaines ou plus -> Phishing (1)
    ext = tldextract.extract(url)
    subdomain = ext.subdomain  # ex: "secure.login.verify.paypal" ou "www"

    # on retire le "www." car c'est un sous-domaine standard, pas suspect
    if subdomain.startswith('www.'):
        subdomain = subdomain[4:]  # supprime les 4 premiers caracteres "www."
    elif subdomain == 'www':
        subdomain = ''  # pas de sous-domaine reel

    # nb_dots = nombre de "morceaux" de sous-domaine
    # subdomain.count('.') compte les points + 1 si sous-domaine non vide
    nb_dots = subdomain.count('.') + (1 if subdomain else 0)

    if nb_dots <= 1:
        return -1
    elif nb_dots == 2:
        return 0
    else:
        return 1


def https_feature(url):
    # Feature HTTPS : verifie si l'URL utilise le protocole HTTPS.
    # La regle officielle UCI verifie aussi l'emetteur du certificat et son age,
    # ce qu'on ne peut pas faire sans appel a une API externe.
    # On simplifie donc : HTTPS present -> probablement legitime.
    #
    # Limite connue : les phisheurs utilisent de plus en plus HTTPS
    # avec des certificats gratuits (Let's Encrypt) pour paraitre credibles.
    # Un vrai audit de certificat serait plus precis mais hors de portee ici.
    #
    # Convention simplifiee :
    #   https:// -> Legitime (-1)
    #   http://  -> Phishing (1)
    return -1 if url.startswith('https://') else 1


def non_std_port(url):
    # Feature NonStdPort : detecte si un port non standard est precise dans l'URL.
    # Les ports standards sont 80 (HTTP) et 443 (HTTPS).
    # Si un autre port est precise explicitement, c'est inhabituel et suspect.
    # Ex: http://site.com:8080/login -> port 8080 = non standard = suspect
    #
    # Regle officielle UCI :
    #   port inhabituel precise -> Phishing (1)
    #   sinon -> Legitime (-1)
    match = re.search(r'://[^/]+:(\d+)', url)
    # ://[^/]+ = apres :// on cherche des caracteres (pas de /) -> c'est le domaine
    # :(\d+) = suivi d'un : et d'un ou plusieurs chiffres -> c'est le port
    if match:
        port = int(match.group(1))  # recupere le numero de port capture
        if port not in (80, 443):   # si c'est pas 80 ni 443 -> non standard
            return 1
    return -1


def https_domain_url(url):
    # Feature HTTPSDomainURL : detecte si le mot "https" apparait
    # dans le NOM DE DOMAINE lui-meme (pas dans le protocole).
    # Technique trompeuse : les phisheurs mettent "https" dans le domaine
    # pour rassurer la victime qui lit vite.
    # Ex: http://https-paypal-secure.com -> contient "https" dans le domaine
    #     mais le protocole est HTTP (pas securise)
    #
    # Regle officielle UCI :
    #   "https" dans le nom de domaine -> Phishing (1)
    #   sinon -> Legitime (-1)
    ext = tldextract.extract(url)
    return 1 if 'https' in ext.domain.lower() else -1


def info_email(url):
    # Feature InfoEmail : detecte la presence de "mailto:" dans l'URL.
    # Les phisheurs utilisent parfois "mailto:" pour rediriger les informations
    # soumises par l'utilisateur directement vers leur boite email.
    # Ex: <form action="mailto:hacker@evil.com"> dans le HTML
    #
    # Regle officielle UCI :
    #   "mailto:" present -> Phishing (1)
    #   sinon -> Legitime (-1)
    return 1 if 'mailto:' in url.lower() else -1


def favicon(url, html=None):
    # Feature Favicon : detecte si l'icone du site (favicon) est chargee
    # depuis un domaine DIFFERENT de la page principale.
    # Un phisheur copie une page legitime mais charge le favicon depuis
    # le vrai site (domaine different) -> trahit la copie.
    # Ex: le favicon de paypal-fake.com est charge depuis paypal.com
    #
    # Regle officielle UCI :
    #   favicon charge depuis un domaine externe -> Phishing (1)
    #   favicon du meme domaine -> Legitime (-1)
    #   pas de HTML disponible -> Legitime (-1) par defaut (valeur neutre)
    if html is None:
        return -1

    ext_site = tldextract.extract(url)  # domaine du site analyse

    # expression reguliere pour trouver <link rel="icon" href="...">
    matches = re.findall(
        r'<link[^>]*rel=["\']icon["\'][^>]*href=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE  # insensible a la casse (Icon, ICON, icon...)
    )

    if not matches:
        return -1  # pas de favicon trouve -> pas de risque pour cette feature

    for href in matches:
        ext_favicon = tldextract.extract(href)
        # si le domaine du favicon est different du domaine du site -> suspect
        if ext_favicon.domain and ext_favicon.domain != ext_site.domain:
            return 1

    return -1  # tous les favicons sont du meme domaine -> legitime


def domain_reg_len(url):
    # Feature DomainRegLen : verifie la duree d'enregistrement du domaine
    # via une requete WHOIS (base de donnees des proprietes de domaines).
    # Les phisheurs enregistrent des domaines pour de courtes durees (< 1 an)
    # car ils savent que le domaine sera signale rapidement.
    # Les sites legitimes paient souvent plusieurs annees a l'avance.
    #
    # Regle officielle UCI :
    #   duree <= 1 an -> Phishing (1)
    #   duree > 1 an -> Legitime (-1)
    #   WHOIS echoue (domaine introuvable ou timeout) -> Suspect (0)
    try:
        import whois
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"  # ex: "google.com"
        info = whois.whois(domain)             # requete WHOIS (peut prendre 1-3s)

        creation = info.creation_date    # date de creation du domaine
        expiration = info.expiration_date # date d'expiration du domaine

        # whois peut retourner une liste de dates au lieu d'une seule
        # (certains registrars retournent plusieurs dates) -> on prend la premiere
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]

        if creation and expiration:
            duree_jours = (expiration - creation).days  # duree totale en jours
            return 1 if duree_jours <= 365 else -1      # <= 1 an = suspect
        return 0  # infos incompletes -> suspect

    except Exception:
        return 0  # WHOIS a echoue -> domaine non verifiable -> suspect


def age_of_domain(url):
    # Feature AgeofDomain : verifie l'age du domaine depuis sa creation.
    # Les phisheurs creent des domaines tres recents (quelques jours/semaines).
    # Les sites legitimes ont generalement plus de 6 mois d'existence.
    #
    # Regle officielle UCI :
    #   age >= 6 mois (180 jours) -> Legitime (-1)
    #   age < 6 mois -> Phishing (1)
    #   WHOIS echoue -> Suspect (0)
    try:
        import whois
        from datetime import datetime

        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        info = whois.whois(domain)

        creation = info.creation_date
        if isinstance(creation, list):
            creation = creation[0]  # on prend la premiere date si c'est une liste

        if creation:
            # calcule le nombre de jours depuis la creation jusqu'a aujourd'hui
            age_jours = (datetime.now() - creation).days
            return -1 if age_jours >= 180 else 1  # >= 180 jours = 6 mois = legitime
        return 0  # date de creation inconnue -> suspect

    except Exception:
        return 0  # WHOIS a echoue -> suspect


def abnormal_url(url):
    # Feature AbnormalURL : verifie si le nom d'hote du domaine est resolvable.
    # La regle officielle UCI verifie via WHOIS que l'identite du proprietaire
    # correspond a ce qui est affiche dans l'URL.
    # On simplifie : on verifie juste que le domaine a un enregistrement DNS valide.
    # Un domaine completement inresolvable est tres probablement du phishing.
    #
    # Regle officielle UCI (approximee) :
    #   domaine non resolvable en DNS -> Phishing (1)
    #   domaine resolvable -> Legitime (-1)
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    try:
        socket.gethostbyname(domain)  # tente de resoudre le domaine en adresse IP
        return -1  # succes -> le domaine existe -> legitime
    except (socket.gaierror, socket.timeout):
        return 1   # echec -> domaine introuvable -> phishing probable


def dns_recording(url):
    # Feature DNSRecording : verifie si le domaine a un enregistrement DNS actif.
    # Tres similaire a AbnormalURL mais avec une logique legerement differente
    # dans la specification UCI (deux features distinctes dans le dataset original).
    # Les sites de phishing ephemeres ont souvent des domaines sans DNS valide
    # car ils sont crees et supprimes tres rapidement.
    #
    # Regle officielle UCI :
    #   pas d'enregistrement DNS -> Phishing (1)
    #   enregistrement DNS trouve -> Legitime (-1)
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    try:
        socket.gethostbyname(domain)  # meme logique que abnormal_url
        return -1  # domaine resolvable -> legitime
    except (socket.gaierror, socket.timeout):
        return 1   # domaine non resolvable -> phishing probable