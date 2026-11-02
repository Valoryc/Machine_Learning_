# 1. DOMANDA INIZIALE
#In questo quesito, noi andremo ad utilizzare un apprendimento Supervisionato,
#poichè il dataset  ha già al suo interno tutto ciò che ci serve, il tipo di problema
#è di classificazione binaria, per il semplice fatto che distinguiamo solo due classi:
#0 : tumore maligno
#1 : tumore benigno 
#In questo caso la colonna target è quella che contiene il risultato della diagnosi.


# IMPORTAZIONI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report


# 2. CARICAMENTO DEL DATASET
cancer = load_breast_cancer(as_frame=True)
X = cancer.data
y = cancer.target

print("Dimensioni:", X.shape)
print("------------------------")
print(X.describe())
print("------------------------")
print("Valori minimi:\n", X.min())
print("------------------------")
print("Valori massimi:\n", X.max())


# 3. DIVISIONE TRAIN, VALIDATION E TEST
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# 4. MODELLO BASELINE
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)

print("Accuracy Dummy:", accuracy_score(y_test, dummy.predict(X_test)))

print("------------------------")

# 5. KNN SENZA SCALING
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Accuracy KNN:", accuracy_score(y_test, knn.predict(X_test)))

print("------------------------")

# 6. SCALING DEI DATI
scaler = StandardScaler() #standardizzazione dei dati
X_train_std = scaler.fit_transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

scaler_minmax = MinMaxScaler() #normalizzazione dei dati
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_val_minmax = scaler_minmax.transform(X_val)
X_test_minmax = scaler_minmax.transform(X_test)

# KNN con StandardScaler
knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_std, y_train)

print("Accuracy KNN StandardScaler:", accuracy_score(y_val, knn_std.predict(X_val_std)))

print("------------------------")

# KNN con MinMaxScaler
knn_minmax = KNeighborsClassifier(n_neighbors=5)
knn_minmax.fit(X_train_minmax, y_train)

print("Accuracy KNN MinMaxScaler:", accuracy_score(y_val, knn_minmax.predict(X_val_minmax)))

print("------------------------")

# 7. SCELTA DEL MIGLIOR K
risultati = []

for k in range(1, 42, 2):
    modello = KNeighborsClassifier(n_neighbors=k)
    modello.fit(X_train_std, y_train)
    predizioni = modello.predict(X_val_std)
    risultati.append(accuracy_score(y_val, predizioni))

miglior_k = list(range(1, 42, 2))[np.argmax(risultati)]
print("Miglior k:", miglior_k)

print("------------------------")

plt.plot(range(1, 42, 2), risultati, marker="o")
plt.xlabel("Numero di vicini (k)")
plt.ylabel("Accuracy")
plt.show()


# 8. ALBERO DECISIONALE
albero = DecisionTreeClassifier(random_state=42)
albero.fit(X_train, y_train)

print("Accuracy albero:", accuracy_score(y_val, albero.predict(X_val)))
print(export_text(albero, feature_names=list(X.columns)))

plot_tree(albero, feature_names=X.columns, class_names=cancer.target_names, filled=True, rounded=True)
plt.show()

print("------------------------")

# Calcolo dell'indice di Gini alla radice
prob = np.bincount(y_train) / len(y_train)
gini = 1 - np.sum(prob ** 2)
print("Gini:", gini)

print("------------------------")

# 9. SCELTA E VALUTAZIONE DEL MODELLO
# KNN con il miglior k
knn_finale = KNeighborsClassifier(n_neighbors=miglior_k)
knn_finale.fit(X_train_std, y_train)

# Confronto dei modelli sul validation set
pred_knn = knn_finale.predict(X_val_std)
pred_albero = albero.predict(X_val)

recall_knn = recall_score(y_val, pred_knn, pos_label=0)
recall_albero = recall_score(y_val, pred_albero, pos_label=0)

print("Recall KNN:", recall_knn)

print("------------------------")

print("Recall albero:", recall_albero)

# Scelta del modello con recall maggiore
if recall_knn >= recall_albero:
    modello_scelto = knn_finale
    X_train_finale = X_train_std
    X_test_finale = X_test_std
    print("------------------------")
    
    print("Modello scelto: KNN")
else:
    modello_scelto = albero
    X_train_finale = X_train
    X_test_finale = X_test
    print("------------------------")
    
    print("Modello scelto: Albero")

# Riaddestramento e valutazione finale
modello_scelto.fit(X_train_finale, y_train)
pred_test = modello_scelto.predict(X_test_finale)

print("------------------------")
print("Accuracy finale:", accuracy_score(y_test, pred_test))

print("------------------------")

print("Recall finale:", recall_score(y_test, pred_test, pos_label=0))
print(classification_report(y_test, pred_test, target_names=cancer.target_names))

print("------------------------")

# Matrice di confusione
cm = confusion_matrix(y_test, pred_test)
print("Matrice di confusione:\n", cm)
print("------------------------")

tn, fp, fn, tp = cm.ravel()
print("TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
