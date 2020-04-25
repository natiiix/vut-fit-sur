Rozpoznávání obličejů (facial_recognition)
==========================================

Klasifikátor je realizován jako konvoluční neuronová síť vytvořená za pomoci knihovny Keras v jazyce Python (verze 3.6.9).
Nástrojem Keras Tuner byl vybrán nejlepší model sítě a ten byl použit ke klasifikaci testovacích dat.
Použitý model je popsán v souboru model_shape, který je přiložen u zdrojového kódu.
Pokud by nastal problém s vytvářením modelu při reprodukci výsledků, je model dostupný v GitHub repozitáři na adrese: https://github.com/natiiix/vut-fit-sur/blob/master/facial_recognition/best_model.h5


Rozpoznávání hlasů metodou K-NN (audio_knn)
===========================================

První verze klasifikátoru hlasu je implementována v jazyce Python (vyvíjeno na verzi 3.7.7) s použitím knihoven NumPy a SciPy.
Vzhledem k nízkému množství trénovacích dat bylo pro algoritmus K-Nearest Neighbors zvoleno K = 3, s nimž bylo dosahováno nejstabilnějších výsledků.
Trénovací data (4 složky, které nám byly poskytnuty se zadáním projektu) je možné vložit do adresáře ~/data/.
Pokud je přítomen soubor ~/known_wavs.pickle, nejsou trénovací data nutná. Tento soubor obsahuje jejich předzpracovanou verzi.
Audio soubory musí být umístěny ve složce ~/data/eval/ a to ve formátu WAV bez jakékoliv hlubší adresářové struktury.
Závislosti jsou specifikovány v souboru ~/requirements.txt.
Samotná logika je implementována ve skriptu ~/__main__.py.
Všechny cesty jsou relativní vůči hlavnímu adresáři daného klasifikátoru, ne domovskému adresáři uživatele.


Rozpoznávání hlasů metodou GMM (audio_gmm)
===========================================

Po uspesnem naimportovani knihoven/modulu numpy, io, ikrlib dojde k nacteni dat na trenovani a dat na naslednou evaluaci.
Extrakce priznaku ze zvukovych nahravek s priponou *.wav je provedena za pomoci funkce z knihovny ikrlib ktera nam vytvori za pomoci segmentace,
Fourieroveho spektra, banky filtru, logaritmizace a MFCC slovnik obsahujici dvojici nazvu souboru a matice hodnot.
Nasledne budou matice slouceny do jednoho velkeho seznamu.
Po nastaveni promennych obou trid target, non_target (napr. apriorni pravdepodobnosti tridy, pocet komponent GMM tridy, stredni hodnoty, kovariancni matici, vahy komponent) se provede N pocet iteraci samotneho trenovani,
kde se s maximalni verohodnosti urci nove hodnoty jiz drive zminenych promennych, ktere by mely zapricinit presnejsi odhad gaussovek.
Ted uz jen zbyva vyhodnotit evaluacni data s jiz natrenovanym GM modelem a vypsat vysledek do souboru.

Doporucuji spustit na novejsi verzi windows s python3.8.2
