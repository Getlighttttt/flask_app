import torch
import random
import matplotlib.pyplot as plt

def initialize_centroids(X, k):
    """
    Vyberie `k` počiatočných centroidov z datasetu.

    Parametre:
    X (torch.Tensor): Dataset v tvare (n, d), kde n je počet dátových bodov
                      a d je počet čŕt (features).
    k (int): Počet zhlukov

    Návratová hodnota:
    torch.Tensor: Tensor s tvarom (k, d), ktorý obsahuje vybrané počiatočné centroidy.
    """
    indices = torch.randperm(X.shape[0])[:k]  # Náhodne vyberieme k indexov
    
    return X[indices]


def compute_distances(X, centroids):
    """
    Vypočíta štvorcovú Euklidovskú vzdialenosť medzi každým dátovým bodom a každým centroidom.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d), kde n je počet dátových bodov
                      a d je počet čŕt (features).
    centroids (torch.Tensor): Centroidy s tvarom (k, d), kde k je počet klastrov.

    Návratová hodnota:
    torch.Tensor: Tensor s tvarom (n, k), obsahujúci štvorcové Euklidovské vzdialenosti.
    """
    
    X_norm = (X ** 2).sum(dim=1, keepdim=True)  # Štvorcová norma každého bodu (n, 1)
    C_norm = (centroids ** 2).sum(dim=1)  # Štvorcová norma každého centroidu (k,)
    distances = X_norm + C_norm - 2 * X @ centroids.T  # Matica štvorcových vzdialeností (n, k)
    
    return distances


def assign_clusters(X, centroids):
    """
    Priradí každý dátový bod k najbližšiemu centroidu.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d).
    centroids (torch.Tensor): Centroidy s tvarom (k, d).

    Návratová hodnota:
    torch.Tensor: Tensor s tvarom (n,), ktorý obsahuje index klastru pre každý dátový bod.
    """
    distances = compute_distances(X, centroids)  # Výpočet vzdialeností (n, k)
    return torch.argmin(distances, dim=1)  # Nájde najbližší centroid pre každý bod


def update_centroids(X, clusters, k, centroids):
    """
    Vypočíta nové centroidy ako priemery bodov priradených ku každému klastru.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d).
    clusters (torch.Tensor): Tensor s tvarom (n,), obsahujúci priradenie klastrov.
    k (int): Počet klastrov.

    Návratová hodnota:
    torch.Tensor: Tensor s tvarom (k, d), obsahujúci aktualizované centroidy.
    """
    new_centroids = torch.zeros((k, X.shape[1]), device=X.device)  # Inicializujeme nové centroidy
    
    for i in range(k):
        cluster_points = X[clusters == i]  # Vyberieme body patriace do klastru i
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(dim=0)  # Vypočítame priemer, ak nie je prázdny
        else:

            new_centroids[i] = X[torch.randint(0, X.shape[0], (1,))]  # Náhodne priradíme nový centroid, ak je prázdny
    
    return new_centroids


def kmeans(X, k=None, max_iters=100, tol=1e-6, device="cpu"):
    """
    Spustí algoritmus K-Means.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d).
    k (int, voliteľné): Počet klastrov.
    max_iters (int, voliteľné): Maximálny počet iterácií (predvolené: 100).
    tol (float, voliteľné): Prahová hodnota konvergencie; algoritmus sa zastaví, ak posun centroidov
                            je menší ako táto hodnota (predvolené: 1e-4).
    device (str, voliteľné): Zariadenie pre výpočty ('cpu' alebo 'cuda') (predvolené: 'cpu').

    Návratová hodnota:
    tuple:
        - torch.Tensor: Tensor s tvarom (n,), obsahujúci konečné priradenia klastrov.
        - torch.Tensor: Tensor s tvarom (k, d), obsahujúci konečné polohy centroidov.
    """
    if k is None:
        k = random.randint(1, 100)  # Náhodne vyberieme k, ak nie je zadané
    
    centroids = initialize_centroids(X, k).to(device)  # Inicializujeme centroidy
    
    for i in range(max_iters):
        clusters = assign_clusters(X, centroids)  # Priradíme body ku klastrom
        new_centroids = update_centroids(X, clusters, k, centroids)  # Vypočítame nové centroidy
        
        shift = torch.norm(new_centroids - centroids)  # Vypočítame posun centroidov
        if shift < tol:
            # print(f"Konvergované po {i+1} iteráciách.")  # Ak sa centroidy už takmer nehýbu, skončíme
            break
        
        centroids = new_centroids  # Aktualizujeme centroidy
    
    return clusters, centroids  # Vrátime konečné klastre a centroidy

def compute_inertia(X, clusters, centroids):
    """
    Vypočíta WCSS (Within-Cluster Sum of Squares), známe aj ako inertia.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d).
    clusters (torch.Tensor): Tensor s tvarom (n,), obsahujúci priradenie klastrov.
    centroids (torch.Tensor): Tensor s tvarom (k, d), obsahujúci centroidy.

    Návratová hodnota:
    float: Celková hodnota WCSS (inertia).
    """
    inertia = 0
    for i in range(centroids.shape[0]):  # Prechádzame cez všetky klastry
        cluster_points = X[clusters == i]  # Vyberieme body patriace do klastru i
        inertia += ((cluster_points - centroids[i]) ** 2).sum()  # Suma štvorcových vzdialeností
    return inertia.item()  # Konvertujeme na číslo pre lepšiu interpretáciu


def elbow_method(X, max_k=None, device="cpu", tol=1500):
    """
    Implementuje metódu lakťa (Elbow Method) na výber optimálneho počtu klastrov.

    Parametre:
    X (torch.Tensor): Dataset s tvarom (n, d).
    max_k (int, voliteľné): Maximálny počet klastrov na testovanie (predvolené: 10).

    Výstup:
    Graf ukazujúci závislosť medzi WCSS a počtom klastrov.
    """
    wcss_values = []
    k = 0
    if max_k == None:
        max_k = X.shape[0]

        for k in range(1, max_k + 1):  # Testujeme rôzne hodnoty k
            clusters, centroids = kmeans(X, k, device=device)
            wcss = compute_inertia(X, clusters, centroids)
            wcss_values.append(wcss)  # Uložíme hodnotu WCSS pre aktuálne k
            
            optimal, found = find_optimal_k(wcss_values, tol)
            if found:
                break
    else:
        for k in range(1, max_k + 1):  # Testujeme rôzne hodnoty k
            clusters, centroids = kmeans(X, k, device=device)
            wcss = compute_inertia(X, clusters, centroids)
            wcss_values.append(wcss)  # Uložíme hodnotu WCSS pre aktuálne k 
            
    optimal, found = find_optimal_k(wcss_values, tol)
    
    return optimal, wcss_values, max_k, k

def find_optimal_k(wcss_values, tol=1050):
    """
    Nájde optimálne k pomocou WCSS (inertia) s danou toleranciou.

    Parametre:
    wcss_values (list): Zoznam hodnôt WCSS pre rôzne k.
    tol (float, voliteľné): Prahová hodnota zlepšenia (predvolené: 1000).

    Návratová hodnota:
    int: Optimálny počet klastrov (k).
    """
    for k in range(1, len(wcss_values)):
        wcss_diff = wcss_values[k - 1] - wcss_values[k]  # Rozdiel medzi WCSS pre k a k+1
        if (wcss_diff < tol and wcss_diff >= 0) or wcss_values[k] == 0.0:
            return k, True  # Prvé k, kde zlepšenie je menšie ako tol
    
    return len(wcss_values), False  # Ak nenájdeme, vrátime max. testované k
