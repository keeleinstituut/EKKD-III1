#Kood, mis võrdleb mudeli märgendeid ÜSi märgenditega
#Autor: Eleri Aedmaa

import pandas as pd
from collections import Counter

# Loe CSV fail
df = pd.read_csv('Claude_m2rgendid.csv', sep=';', encoding='cp1252')

print("VEERGUDE NIMED FAILIS:")
print("=" * 80)
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")
print("=" * 80)
print()

# Funktsioon märgendite võrdlemiseks
def vorda_margendeid_korduvusega(mudel_margendid, us_margendid):
    """
    Võrdleb mudeli märgendeid ÜSi märgenditega
    ARVESTAB MÄRGENDITE KORDUVUST
    
    Tagastab dictionary kõigi komponentidega eraldi
    """
    # Eemalda tühjad ja None väärtused, aga SÄILITA korduvus
    mudel_list = [str(m).strip() for m in mudel_margendid if pd.notna(m) and str(m).strip()]
    us_list = [str(m).strip() for m in us_margendid if pd.notna(m) and str(m).strip()]
    
    # Kasuta Counter'it korduvuse jaoks
    mudel_counter = Counter(mudel_list)
    us_counter = Counter(us_list)
    
    # Leia unikaalsed märgendid (set'id)
    mudel_set = set(mudel_counter.keys())
    us_set = set(us_counter.keys())
    
    # Arvuta erinevused (märgendite tüübid)
    uhised_margendid = mudel_set.intersection(us_set)
    ainult_mudel = mudel_set - us_set
    ainult_us = us_set - mudel_set
    
    # Arvuta korduvuste võrdlus ühistele märgenditele
    korduvuste_vordlus = {}
    for marg in uhised_margendid:
        korduvuste_vordlus[marg] = {
            'mudel': mudel_counter[marg],
            'us': us_counter[marg],
            'vahe': mudel_counter[marg] - us_counter[marg]
        }
    
    # Määra kategooria
    if len(mudel_list) == 0:
        kategooria = 'ei'
        selgitus = 'Mudel ei pakkunud ühtegi märgendit'
    elif len(us_list) == 0:
        kategooria = 'ei'
        selgitus = 'ÜSis ei ole märgendeid võrrelda'
    elif len(uhised_margendid) == 0:
        kategooria = 'ei'
        selgitus = 'Ükski märgend ei klapi'
    elif mudel_counter == us_counter:
        kategooria = 'jah_100'
        selgitus = '100% klapp (ka korduvused)'
    elif mudel_set == us_set:
        # Kõik märgendid on olemas, aga korduvused erinevad
        kategooria = 'jah_korduvused_erinevad'
        selgitus = 'Kõik märgendid olemas, aga korduvused erinevad'
    elif us_set.issubset(mudel_set):
        kategooria = 'jah_plus'
        selgitus = 'Kõik ÜSi märgendid olemas + lisandused'
    else:
        kategooria = 'osaliselt'
        selgitus = 'Osa märgendeid klapib'
    
    # Formateeri märgendid koos kordusvustega
    def format_with_counts(counter):
        if not counter:
            return ''
        return ', '.join([f"{marg}({count})" if count > 1 else marg 
                         for marg, count in sorted(counter.items())])
    
    return {
        'kategooria': kategooria,
        'selgitus': selgitus,
        
        # Unikaalsed märgendid
        'uhised': ', '.join(sorted(uhised_margendid)) if uhised_margendid else '',
        'ainult_mudel': ', '.join(sorted(ainult_mudel)) if ainult_mudel else '',
        'ainult_us': ', '.join(sorted(ainult_us)) if ainult_us else '',
        
        # Arvud
        'uhiste_arv': len(uhised_margendid),
        'ainult_mudel_arv': len(ainult_mudel),
        'ainult_us_arv': len(ainult_us),
        
        # UUED: Korduvuse statistika
        'mudel_kokku': len(mudel_list),
        'us_kokku': len(us_list),
        'mudel_unikaalseid': len(mudel_set),
        'us_unikaalseid': len(us_set),
        
        # Märgendid koos korduvustega
        'mudel_korduvustega': format_with_counts(mudel_counter),
        'us_korduvustega': format_with_counts(us_counter),
        
        # Korduvuste võrdlus ühistele märgenditele
        'korduvuste_vordlus': '; '.join([f"{m}: mudel={v['mudel']}, ÜS={v['us']}" 
                                         for m, v in sorted(korduvuste_vordlus.items())])
    }


# Grupeeri andmed katsesõnade kaupa
tulemused = []

# Leia veeru nimi, mis sisaldab "ÜSis" või "�sis"
us_veerg = None
for col in df.columns:
    if 'sis' in col.lower() and len(col) < 10:
        us_veerg = col
        break

if not us_veerg:
    print("VIGA: Ei leidnud ÜSi märgendite veergu!")
    exit(1)

print(f"Kasutan järgmisi veerge:")
print(f"  ÜS märgendid: '{us_veerg}'")
print(f"  Märgend_korp: 'Märgend_korp'")
print(f"  Märgend_treening: 'Märgend_treening'")
print()

for sona in df['katsesõna'].unique():
    if pd.isna(sona):
        continue
    
    sona_read = df[df['katsesõna'] == sona]
    
    # Kogume märgendid KORPUS veerust (Märgend_korp)
    # SÄILITAME KÕIK, KA DUPLIKAADID!
    margend_korp = []
    for idx, row in sona_read.iterrows():
        if pd.notna(row.get('Märgend_korp')):
            margend_korp.append(row['Märgend_korp'])
    
    # Kogume märgendid TREENING veerust (Märgend_treening) - PARANDATUD!
    margend_treening = []
    for idx, row in sona_read.iterrows():
        if pd.notna(row.get('Märgend_treening')):
            margend_treening.append(row['Märgend_treening'])
    
    # Kogume kõik ÜSi märgendid (SÄILITAME KORDUVUSE!)
    us_margendid = []
    for idx, row in sona_read.iterrows():
        m = row[us_veerg]
        if pd.notna(m):
            if ',' in str(m):
                us_margendid.extend([x.strip() for x in str(m).split(',')])
            else:
                us_margendid.append(str(m).strip())
    
    # Võrdleme mõlemat SKM märgendi veergu eraldi ÜSi märgenditega
    vordlus_korp = vorda_margendeid_korduvusega(margend_korp, us_margendid)
    vordlus_treening = vorda_margendeid_korduvusega(margend_treening, us_margendid)
    
    tulemused.append({
        'katsesõna': sona,
        
        # ÜS
        'ÜS_märgendid': vordlus_korp['us_korduvustega'],
        'ÜS_kokku': vordlus_korp['us_kokku'],
        'ÜS_unikaalseid': vordlus_korp['us_unikaalseid'],
        
        # KORPUS
        'SKM_korp_märgendid': vordlus_korp['mudel_korduvustega'],
        'SKM_korp_kokku': vordlus_korp['mudel_kokku'],
        'SKM_korp_unikaalseid': vordlus_korp['mudel_unikaalseid'],
        'SKM_korp_kategooria': vordlus_korp['kategooria'],
        'SKM_korp_selgitus': vordlus_korp['selgitus'],
        'SKM_korp_ühised': vordlus_korp['uhised'],
        'SKM_korp_ainult_mudel': vordlus_korp['ainult_mudel'],
        'SKM_korp_ainult_ÜS': vordlus_korp['ainult_us'],
        'SKM_korp_korduvuste_võrdlus': vordlus_korp['korduvuste_vordlus'],
        
        # TREENING
        'SKM_treening_märgendid': vordlus_treening['mudel_korduvustega'],
        'SKM_treening_kokku': vordlus_treening['mudel_kokku'],
        'SKM_treening_unikaalseid': vordlus_treening['mudel_unikaalseid'],
        'SKM_treening_kategooria': vordlus_treening['kategooria'],
        'SKM_treening_selgitus': vordlus_treening['selgitus'],
        'SKM_treening_ühised': vordlus_treening['uhised'],
        'SKM_treening_ainult_mudel': vordlus_treening['ainult_mudel'],
        'SKM_treening_ainult_ÜS': vordlus_treening['ainult_us'],
        'SKM_treening_korduvuste_võrdlus': vordlus_treening['korduvuste_vordlus']
    })

# Loo tulemuste DataFrame
tulemused_df = pd.DataFrame(tulemused)

# Salvesta tulemused (kasutame tabulaatorit eraldajana)
tulemused_df.to_csv('margendite_vordlus_tulemused_Claude.csv', index=False, sep='\t', encoding='utf-8-sig')

# Prindi kokkuvõte
print("=" * 80)
print("MÄRGENDITE VÕRDLUSE KOKKUVÕTE (KORDUVUSEGA!)")
print("=" * 80)
print()

print("SKM KORPUS (Märgend_korp) - KATEGOORIATE JAOTUS:")
print("-" * 40)
kategooriad_korp = tulemused_df['SKM_korp_kategooria'].value_counts()
for kat, arv in kategooriad_korp.items():
    protsent = (arv / len(tulemused_df)) * 100
    if kat == 'jah_100':
        print(f"✓ JAH (100% klapp + korduvused): {arv} ({protsent:.1f}%)")
    elif kat == 'jah_korduvused_erinevad':
        print(f"≈ JAH (märgendid OK, korduvused erinevad): {arv} ({protsent:.1f}%)")
    elif kat == 'jah_plus':
        print(f"✓ JAH (+ lisandused): {arv} ({protsent:.1f}%)")
    elif kat == 'osaliselt':
        print(f"≈ OSALISELT: {arv} ({protsent:.1f}%)")
    elif kat == 'ei':
        print(f"✗ EI: {arv} ({protsent:.1f}%)")

print()
print("SKM TREENING (Märgend_treening) - KATEGOORIATE JAOTUS:")
print("-" * 40)
kategooriad_treening = tulemused_df['SKM_treening_kategooria'].value_counts()
for kat, arv in kategooriad_treening.items():
    protsent = (arv / len(tulemused_df)) * 100
    if kat == 'jah_100':
        print(f"✓ JAH (100% klapp + korduvused): {arv} ({protsent:.1f}%)")
    elif kat == 'jah_korduvused_erinevad':
        print(f"≈ JAH (märgendid OK, korduvused erinevad): {arv} ({protsent:.1f}%)")
    elif kat == 'jah_plus':
        print(f"✓ JAH (+ lisandused): {arv} ({protsent:.1f}%)")
    elif kat == 'osaliselt':
        print(f"≈ OSALISELT: {arv} ({protsent:.1f}%)")
    elif kat == 'ei':
        print(f"✗ EI: {arv} ({protsent:.1f}%)")

print()
print(f"Kokku analüüsitud: {len(tulemused_df)} katsesõna")
print()

# LISAME STATISTIKA MÄRGENDITE ARVU KOHTA
print("=" * 80)
print("DETAILNE STATISTIKA (KORDUVUSTEGA)")
print("=" * 80)
print()

print("ÜS:")
print(f"  Keskmine märgendite arv kokku: {tulemused_df['ÜS_kokku'].mean():.2f}")
print(f"  Keskmine unikaalsete märgendite arv: {tulemused_df['ÜS_unikaalseid'].mean():.2f}")
print()

print("KORPUS:")
print(f"  Keskmine märgendite arv kokku: {tulemused_df['SKM_korp_kokku'].mean():.2f}")
print(f"  Keskmine unikaalsete märgendite arv: {tulemused_df['SKM_korp_unikaalseid'].mean():.2f}")
print()

print("TREENING:")
print(f"  Keskmine märgendite arv kokku: {tulemused_df['SKM_treening_kokku'].mean():.2f}")
print(f"  Keskmine unikaalsete märgendite arv: {tulemused_df['SKM_treening_unikaalseid'].mean():.2f}")
print()

# Detailne väljund (esimesed 10)
print("=" * 80)
print("DETAILSED TULEMUSED (esimesed 10)")
print("=" * 80)
print()

for idx, row in tulemused_df.head(10).iterrows():
    print(f"{idx + 1}. {row['katsesõna'].upper()}")
    print(f"   ÜS: {row['ÜS_märgendid']} (kokku: {row['ÜS_kokku']}, unikaalseid: {row['ÜS_unikaalseid']})")
    print()
    print(f"   SKM KORPUS:")
    print(f"      Märgendid: {row['SKM_korp_märgendid']}")
    print(f"      Kokku: {row['SKM_korp_kokku']}, Unikaalseid: {row['SKM_korp_unikaalseid']}")
    print(f"      Kategooria: {row['SKM_korp_kategooria']}")
    print(f"      Ühised: {row['SKM_korp_ühised']}")
    print(f"      Ainult mudel: {row['SKM_korp_ainult_mudel']}")
    print(f"      Ainult ÜS: {row['SKM_korp_ainult_ÜS']}")
    if row['SKM_korp_korduvuste_võrdlus']:
        print(f"      Korduvused: {row['SKM_korp_korduvuste_võrdlus']}")
    print()
    print(f"   SKM TREENING:")
    print(f"      Märgendid: {row['SKM_treening_märgendid']}")
    print(f"      Kokku: {row['SKM_treening_kokku']}, Unikaalseid: {row['SKM_treening_unikaalseid']}")
    print(f"      Kategooria: {row['SKM_treening_kategooria']}")
    print(f"      Ühised: {row['SKM_treening_ühised']}")
    print(f"      Ainult mudel: {row['SKM_treening_ainult_mudel']}")
    print(f"      Ainult ÜS: {row['SKM_treening_ainult_ÜS']}")
    if row['SKM_treening_korduvuste_võrdlus']:
        print(f"      Korduvused: {row['SKM_treening_korduvuste_võrdlus']}")
    print()

print("=" * 80)
print("Tulemused on salvestatud faili: margendite_vordlus_tulemused_Claude.csv")
print("CSV fail kasutab TAB-eraldajat ja sisaldab kõiki detaile eraldi veergudes!")
print("=" * 80)