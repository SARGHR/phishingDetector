import sys
import os

sys.path.append(os.path.dirname(__file__))

from url_features import (
    using_ip, long_url, short_url, symbol_at, redirecting_double_slash,
    prefix_suffix, sub_domains, https_feature, non_std_port,
    https_domain_url, info_email, favicon, abnormal_url,
    dns_recording, domain_reg_len, age_of_domain
)
from html_features import telecharger_page, extract_html_features


def extraire_toutes_les_features(url):
    """Calcule les 30 features dans l'ORDRE EXACT attendu par le modèle.
    Télécharge le HTML une seule fois et le réutilise partout."""

    html = telecharger_page(url)
    html_feats = extract_html_features(url)
    # ⚠️ extract_html_features télécharge déjà la page elle-même en interne,
    # donc on télécharge 2 fois — on corrigera ça juste après.

    features = {
        'UsingIP':           using_ip(url),
        'LongURL':            long_url(url),
        'ShortURL':           short_url(url),
        'Symbol@':            symbol_at(url),
        'Redirecting//':      redirecting_double_slash(url),
        'PrefixSuffix-':      prefix_suffix(url),
        'SubDomains':         sub_domains(url),
        'HTTPS':              https_feature(url),
        'DomainRegLen':       domain_reg_len(url),
        'Favicon':            favicon(url, html),
        'NonStdPort':         non_std_port(url),
        'HTTPSDomainURL':     https_domain_url(url),
        'RequestURL':         html_feats['RequestURL'],
        'AnchorURL':          html_feats['AnchorURL'],
        'LinksInScriptTags':  html_feats['LinksInScriptTags'],
        'ServerFormHandler':  html_feats['ServerFormHandler'],
        'InfoEmail':          info_email(url),
        'AbnormalURL':        abnormal_url(url),
        'WebsiteForwarding':  html_feats['WebsiteForwarding'],
        'StatusBarCust':      html_feats['StatusBarCust'],
        'DisableRightClick':  html_feats['DisableRightClick'],
        'UsingPopupWindow':   html_feats['UsingPopupWindow'],
        'IframeRedirection':  html_feats['IframeRedirection'],
        'AgeofDomain':        age_of_domain(url),
        'DNSRecording':       dns_recording(url),
        'WebsiteTraffic':     html_feats['WebsiteTraffic'],
        'PageRank':           html_feats['PageRank'],
        'GoogleIndex':        html_feats['GoogleIndex'],
        'LinksPointingToPage': html_feats['LinksPointingToPage'],
        'StatsReport':        html_feats['StatsReport'],
    }

    return features


if __name__ == '__main__':
    url = 'https://www.google.com'
    print(f"Test complet sur {url}\n")
    feats = extraire_toutes_les_features(url)
    for i, (k, v) in enumerate(feats.items()):
        print(f"{i}: {k} = {v}")