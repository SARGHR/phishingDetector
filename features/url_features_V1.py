import re
import tldextract

def extract_url_features(url):
    #decomposer l'URL en (subdomain,domain,ext)
    ext = tldextract.extract(url)
    
    #creation de dict pour stocker les info suivantes
    features = {}

    # --- LONGUEUR ---
    features['length'] = len(url)
    # Pourquoi ? Les URLs de phishing sont souvent très longues
    # pour noyer le vrai domaine dans du bruit

    features['n_dots'] = url.count('.')
    # Pourquoi ? Plus de points = plus de sous-domaines suspects
    # ex: secure.paypal.verify.login.malicious.com

    features['n_hyphens'] = url.count('-')
    # Pourquoi ? Les hyphens sont utilisés pour imiter des domaines
    # ex: paypal-secure.com au lieu de paypal.com

    features['n_slashes'] = url.count('/')
    # Pourquoi ? Un chemin très profond est suspect
    # ex: /login/verify/confirm/account/update

    features['n_digits'] = sum(c.isdigit() for c in url)
    # Pourquoi ? Les IPs et URLs générées automatiquement
    # contiennent beaucoup de chiffres

    # --- STRUCTURE ---
    features['has_ip'] = int(bool(
        #motif(\d: un ou plusieurs chiffres)
        re.search(r'\d+\.\d+\.\d+\.\d+', url)
    ))
    # Pourquoi ? Un vrai site utilise un nom de domaine,
    # pas une IP brute comme http://192.168.1.1/login

    features['has_https'] = int(url.startswith('https'))
    # Pourquoi ? Les sites légitimes utilisent HTTPS
    # (mais attention : les phishers aussi de plus en plus)

    features['has_at_symbol'] = int('@' in url)
    # Pourquoi ? Le @ dans une URL force le navigateur à ignorer
    # tout ce qui précède : http://google.com@malicious.com
    # → le navigateur va sur malicious.com

    features['has_double_slash'] = int('//' in url[7:])
    # Pourquoi ? Une redirection cachée
    # ex: http://legit.com//http://malicious.com
    # On commence à chercher après le http:// initial (index 7)

    features['has_prefix_suffix'] = int('-' in ext.domain)
    # Pourquoi ? paypal-secure.com imite paypal.com
    # Le domaine légitime ne contient jamais de tiret

    # --- DOMAINE ---
    features['n_subdomains'] = len(
        [s for s in ext.subdomain.split('.') if s]
    )
    # Pourquoi ? secure.login.verify.paypal.malicious.com
    # a 4 sous-domaines — très suspect

    features['domain_length'] = len(ext.domain)
    # Pourquoi ? Les domaines générés automatiquement
    # sont souvent très longs ou très courts

    suspicious_tlds = [
        '.tk', '.ml', '.ga', '.cf', '.gq',  # gratuits et souvent abusés
        '.xyz', '.top', '.click', '.link'
    ]
    features['suspicious_tld'] = int(
    ('.' + ext.suffix) in suspicious_tlds
    )
    # Pourquoi ? Ces extensions sont gratuites et massivement
    # utilisées par les phishers car sans vérification d'identité

    # --- MOTS SUSPECTS ---
    suspicious_words = [
        'login', 'signin', 'verify', 'secure', 'account',
        'update', 'confirm', 'banking', 'password', 'credential',
        # Versions françaises
        'connexion', 'verification', 'compte', 'suspendu',
        'urgent', 'ameli', 'impots', 'caf'
    ]
    url_lower = url.lower()
    features['n_suspicious_words'] = sum(
        w in url_lower for w in suspicious_words
    )
    # Pourquoi ? Les phishers mettent ces mots dans l'URL
    # pour rassurer la victime : secure-login-verify-paypal.com

    #    re.search(...)  →  objet Match ou None
    #bool(...)       →  True ou False
    #int(...)        →  1 ou 0  ← le ML a besoin de chiffres
    
    features['has_shortener'] = int(bool(re.search(
        r'bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly', url_lower
    )))
    # Pourquoi ? Les raccourcisseurs d'URL cachent la vraie destination
    # et sont très utilisés dans les campagnes de phishing

    return features


# Test rapide si on lance ce fichier directement
if __name__ == '__main__':
    test_urls = [
        'https://www.google.com',
        'http://paypal-secure.verify-login.tk/account/update',
        'http://192.168.1.1/login/confirm'
    ]
    for url in test_urls:
        print(f"\n--- {url} ---")
        feats = extract_url_features(url)
        for k, v in feats.items():
            print(f"  {k}: {v}")