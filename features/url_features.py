import re
import socket
import tldextract


def using_ip(url):
    """UsingIP : IP brute à la place d'un nom de domaine
    Règle officielle : IP présente → Phishing(1), sinon → Legitimate(-1)"""
    ip_pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|(0x[0-9a-fA-F]+)'
    return 1 if re.search(ip_pattern, url) else -1


def long_url(url):
    """LongURL : URL longue = suspect
    Règle officielle : <54 → Legitimate(-1), 54-75 → Suspicious(0), >75 → Phishing(1)"""
    length = len(url)
    if length < 54:
        return -1
    elif length <= 75:
        return 0
    else:
        return 1


def short_url(url):
    """ShortURL : utilise un raccourcisseur connu
    Règle officielle : TinyURL/raccourci → Phishing(1), sinon → Legitimate(-1)"""
    shorteners = r'bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly|is\.gd|buff\.ly'
    return 1 if re.search(shorteners, url.lower()) else -1


def symbol_at(url):
    """Symbol@ : le @ fait ignorer tout ce qui précède au navigateur
    Règle officielle : @ présent → Phishing(1), sinon → Legitimate(-1)"""
    return 1 if '@' in url else -1


def redirecting_double_slash(url):
    """Redirecting// : // après position 7 = redirection cachée
    Règle officielle : position du dernier // > 7 → Phishing(1), sinon → Legitimate(-1)"""
    return 1 if url.rfind('//') > 6 else -1


def prefix_suffix(url):
    """PrefixSuffix- : tiret dans le nom de domaine (paypal-secure.com)
    Règle officielle : tiret présent → Phishing(1), sinon → Legitimate(-1)"""
    ext = tldextract.extract(url)
    return 1 if '-' in ext.domain else -1


def sub_domains(url):
    """SubDomains : nombre de sous-domaines
    Règle officielle : 1 point → Legitimate(-1), 2 points → Suspicious(0), 3+ points → Phishing(1)
    (on compte les points dans le sous-domaine après avoir retiré www.)"""
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    if subdomain.startswith('www.'):
        subdomain = subdomain[4:]
    elif subdomain == 'www':
        subdomain = ''
    nb_dots = subdomain.count('.') + (1 if subdomain else 0)
    if nb_dots <= 1:
        return -1
    elif nb_dots == 2:
        return 0
    else:
        return 1


def https_feature(url):
    """HTTPS : présence du protocole sécurisé
    Simplification : https → Legitimate(-1), http → Phishing(1)
    ⚠️ Limite connue : un vrai audit de certificat serait plus précis (cf rapport)"""
    return -1 if url.startswith('https://') else 1


def non_std_port(url):
    """NonStdPort : un port non-standard est explicitement précisé dans l'URL
    Règle officielle : présence d'un port inhabituel → Phishing(1), sinon → Legitimate(-1)"""
    match = re.search(r'://[^/]+:(\d+)', url)
    if match:
        port = int(match.group(1))
        if port not in (80, 443):
            return 1
    return -1


def https_domain_url(url):
    """HTTPSDomainURL : le mot 'https' est utilisé dans le nom de domaine lui-même
    Règle officielle : présent → Phishing(1), sinon → Legitimate(-1)"""
    ext = tldextract.extract(url)
    return 1 if 'https' in ext.domain.lower() else -1


def info_email(url):
    """InfoEmail : présence de mailto: dans l'URL
    Règle officielle : présent → Phishing(1), sinon → Legitimate(-1)"""
    return 1 if 'mailto:' in url.lower() else -1


def favicon(url, html=None):
    """Favicon : le favicon est chargé depuis un domaine externe
    Règle officielle : externe → Phishing(1), sinon → Legitimate(-1)
    Si pas de HTML disponible, valeur neutre Legitimate(-1) par défaut"""
    if html is None:
        return -1
    ext_site = tldextract.extract(url)
    matches = re.findall(r'<link[^>]*rel=["\']icon["\'][^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if not matches:
        return -1
    for href in matches:
        ext_favicon = tldextract.extract(href)
        if ext_favicon.domain and ext_favicon.domain != ext_site.domain:
            return 1
    return -1


def domain_reg_len(url):
    """DomainRegLen : durée d'enregistrement du domaine (whois)
    Règle officielle : expire dans ≤1 an → Phishing(1), sinon → Legitimate(-1)
    Si le WHOIS échoue (domaine introuvable), c'est suspect → valeur neutre(0)"""
    try:
        import whois
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        info = whois.whois(domain)

        creation = info.creation_date
        expiration = info.expiration_date
        if isinstance(creation, list):
            creation = creation[0]
        if isinstance(expiration, list):
            expiration = expiration[0]

        if creation and expiration:
            duree_jours = (expiration - creation).days
            return 1 if duree_jours <= 365 else -1
        return 0  # infos incomplètes = suspect

    except Exception:
        return 0  # WHOIS échoue = domaine non vérifiable = suspect


def age_of_domain(url):
    """AgeofDomain : âge du domaine depuis sa création
    Règle officielle : ≥6 mois → Legitimate(-1), sinon → Phishing(1)
    Si le WHOIS échoue, c'est suspect → valeur neutre(0)"""
    try:
        import whois
        from datetime import datetime

        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}"
        info = whois.whois(domain)

        creation = info.creation_date
        if isinstance(creation, list):
            creation = creation[0]

        if creation:
            age_jours = (datetime.now() - creation).days
            return -1 if age_jours >= 180 else 1
        return 0

    except Exception:
        return 0


def abnormal_url(url):
    """AbnormalURL : le nom d'hôte n'est pas trouvable/résolvable
    Règle officielle (approximée) : pas de DNS valide → Phishing(1), sinon → Legitimate(-1)"""
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    try:
        socket.gethostbyname(domain)
        return -1
    except (socket.gaierror, socket.timeout):
        return 1


def dns_recording(url):
    """DNSRecording : le domaine a un enregistrement DNS
    Règle officielle : pas de DNS → Phishing(1), sinon → Legitimate(-1)"""
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}"
    try:
        socket.gethostbyname(domain)
        return -1
    except (socket.gaierror, socket.timeout):
        return 1