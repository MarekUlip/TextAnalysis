aliaser - skript sloužící pro kratší názvy volání vrstev z Kerasu
Bidirectional_RNN_analyser - skript pro nalezení dobré architekury u obousměrné rekurentní sítě
Bidirectional_RNN_Embedding_analyser - skript pro nalezení dobré architekury u obousměrné rekurentní sítě s využitím vrstvy vnoření
Conv_GRU_RNN_analyser - skript pro nalezení dobré architektury u kovnoluční sítě využívající GRU
Dense_analyser - skript pro nalezení dobré architektury u hustě propojené sítě
embedding_loader - skript pro načtení matice použité jako váhy pro vrstvu vnoření
GRU_RNN_analyser - skript pro nalezení dobré architektury u GRU sítě
hyperparameter_search - skript sloužící pro nalezení dobrých parametrů, aby sítě doshovali vysoké přesnosti u daného datasetu
KerasRunTester - skript pro ověření, že základní knihovny potřebné pro tento projekt přítomny a funkční.
lda_impl - implementace LDA použitá při porovnávání kvality topiků s neuronovým modelem
LDALoopTests - skript provádějící experimenty s modelem LDA. Většinou byl využit pouze pro získání topiků tímto modelem.
LSTM_GloVe_Embedding_analyser - skript pro nalezení dobré architektury u LSTM sítě využívající vrstvu vnoření inicializovanou pomocí vah GloVe
LSTM_RNN_analyser - skript pro nalezení dobré architektury u LSTM sítě
LSTMEmbedding - skript pro nalezení dobré architektury u LSTM sítě využívající nepředtrénovanou vrstvu vnoření
model_accuracy_tester - skript sloužící pro testování přesnosti jednotlivých architektur na jednotlivých datasetech s daným způsobem předzpracování
model_accuracy_tester_looper - podobný skript jako model_accuracy_tester s tím rozdílem, že zde šlo provádět několik testů po sobě (Občas se stávalo, že při vytvoření více architektur v jednom běhu způsobilo pád skriptu. Avšak po čase se ukázalo, že se jedná pouze o určité architektury a nezávisle na tom, kolik dalších architektur bylo před toutou architekturou spuštěno. Proto byla dodatečně vytvořena tato cyklová alternativa)
NeuralLDAanalysisMethods - metody, které se často používali v různých skriptech experimentujících s neuronovým modelem topiků.
NeuralLDAFitNormalize - skript experimentující s vlivem normalizace vah po každé dávce na kvalitu topiků
NeuralLDALoopTests - skript podobný skriptu NeuralLDATestRegularize akorát s tím rozdílem, že zde je možné provést více testů vlivu regularizace.
NeuralLDATestNormalize - skript experimentující s vlivem použití vrstvy normalizace na kvalitu topiků
NeuralLDATestRegularize - skript experimentující s vlivem různých druhů regularizace na kvalitu topiků
NeuralLDATestRegularizeTFIDF - podobný skript jako NeuralLDATestRegularize s tím rozdílem, že zde byla pro reprezentaci topiků použita matice TFIDF vytvořena pomocí knihovny sklearn namísto použití tokenizátoru
NeuralLDATFIDFFitNormalize - podobný skript jako NeuralLDAFitNormalize s tím rozdílem, že zde byla pro reprezentaci topiků použita matice TFIDF vytvořena pomocí knihovny sklearn namísto použití tokenizátoru
NeuralTopicMatrix - třída reprezentující model topiků vytvořený z vah autoenkodéru (buď vstupní nebo výstupní). Umožňuje také provádět shlukování.
simple_rnn_analyser - skript pro nalezení dobré architektury pro základní rekurentní síť (Vanilla RNN)

