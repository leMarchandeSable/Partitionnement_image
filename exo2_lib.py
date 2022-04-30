import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
from Ma314.BE__Ma314_2021.ACP import correlationdirprinc, cercle_cor


def esperance(Xi):
    """
    fonction qui calcul un estimateur sans biais de l'esperence d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: l'esperence du vecteur Xi
    """

    m = Xi.shape[0]
    return np.sum(Xi) / m


def variance(Xi):
    """
    fonction qui calcul un estimateur avec un biais asymptotique de la variance d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: la variance du vecteur Xi
    """

    m = Xi.shape[0]
    Xi_bar = esperance(Xi)
    return np.sum((Xi - Xi_bar) ** 2) / (m - 1)


def centre_red(R):
    """
    fonction qui a partir une variable aleatoire R de loi differente, centre et reduit les Xi pour qu'ils soient plus
    homogène a etudier
    :param R: un vecteur aléatoire de taille (m, n)
    :return Rcr: le vecteur aléatoire R modifié de facon que l'esperance de Xi soient 0 et leurs variance 1
    """

    # on récupère les dimensions de R pour créer la matrice résultat Rcr de même dimension
    m, n = R.shape
    Rcr = np.zeros((m, n))

    # pour chaque colonne de R on centre et réduit indépendamment les données car les Xi ne suivent pas les mêmes lois
    for i in range(n):
        Xi = R[:, i]

        # on calcule l'espérence et la variance de chaque Xi
        E = esperance(Xi)
        var = variance(Xi)

        # on "décale" les données de chaque colonnes indépendamment
        Rcr[:, i] = (Xi - E) / np.sqrt(var)

    # on retourne la matrice Rcr qui contient les données centrées et réduites
    return Rcr


def barycentre(S):
    # retourne un matrice de taille (k, p) ou chaque ligne est le barycentre de tous les points d'un catégorie S[k]
    return np.array([np.sum(s, axis=0, dtype=float) / len(s) for s in S])


def ACP(X, q, w=None):
    """
    le but de la fonction est de decomposé en vecteur propre la matrice R suivant k direction que l'on doit déterminer,
    apres avoir projeté R sur ces vecteurs la matrice résultante sera dans un EV de dimension plus faible donc
    possiblement affichable sur un plan
    :param X: le vecteur aléatoire, matrice de données de taille (m, n)
    :param q: le nombre de dimension dans lequel on souhaite projeter X
    :return acp: un matrice de taille (m, q)
    """

    # on récupère les dimensions de la matrice R et on créer la matrice proj qui contiendra le résultat
    n, d = X.shape

    # on centre et réduit le vecteur aléatoire R pour que les composantes soient toutes homogènes
    Xcr = centre_red(X)

    if w is None:
        W = np.eye(d)
    else:
        W = np.diag(w)

    C = (Xcr @ np.sqrt(W)).T @ (Xcr @ np.sqrt(W)) / np.sum(W)
    U, S, VT = np.linalg.svd(C)
    acp = Xcr @ U[:, :q]

    kaiser = 0
    while S[kaiser] > np.sum(S) / d:
        kaiser += 1

    return acp, kaiser


def Kmoy(A, k, err=10 ** -15):
    """
        L'algorithme des k-means permette de catégoriser / classifier les objets d'un EV en fonction de leurs distances
        euclidiennes relative. Dans notre cas il permet de séparer en k catégorie un nuage de point que l'on
        sait 'relativement ordonné' grace a un travail de décomposition en vecteur propre ACP effectué précédements.
        :param A: la matrice de taille (m, p) que souhaite classifier
        :param k: le nombre de groupe / catégorie que l'on souhaite créer
        :param err: condition d'arrete de l'algorithme : c'est la variation minimal entre les barycentres muu entre 2
                    ittérations pour stopper l'algorithme
        :return S: la meilleur partition des lignes de A,
                   une liste de longueur k qui contient les lignes de A trier selon k catégories exemple,
                   pour k = 3: S = [[.., .., ..], [.., .., .., .., ..], [.., ..]]
        """

    # on récupère les dimensions de A
    m, p = A.shape

    # ------------------------------------- initialistation de l'algorithme --------------------------------------------
    # on choisie k ligne différente de A au hasard
    # on recommence tant que l'on a pas k indice différent
    indice = np.random.choice(m, size=k)
    while len(np.unique(indice)) != k:
        indice = np.random.choice(m, size=k)

    # muu est un matrice de taille (k, p) qui contient les coordonnées des barycentres des catégories que l'on cherche
    # on initialise ces barycentres de manière aléatoire a partir des indices
    muu = A[indice, :]
    # on fixe delta_muu est notre variable de souvenir (ittération n-1)
    delta_muu = muu
    S = None

    # ------------------------------------------- algorithme k-means ---------------------------------------------------
    # tant que l'algorithme n'est pas stable, que les barycentres se deplacent encore on ittere l'algo
    while np.linalg.norm(delta_muu) > err:
        # on initialise notre partition des lignes de A en k catégorie : [[], [], .., []]
        S = [[] for l in range(k)]

        # pour chaque ai les ligne de A, on clacule sa distance aux k-barycentres et on ajoute ai au nuage de point
        # dont le barycentre est le plus proche
        for i in range(m):
            ai = A[i, :]

            # on calcule les distances entre ai et les k-barycentres avec la norme 2
            D = [np.linalg.norm(ai - muu[j, :]) for j in range(k)]
            # on récupère l'indice du nuage le plus proche de ai, avec j la catégorie qui lui conviendrait le mieux
            j, _ = min(enumerate(D), key=lambda x: x[1])
            # on ajoute la ligne ai a la meilleur catégorie au vu des barycentres de cette ittération
            S[j].append(ai)

        # il se peut qu'un catégorie de S soit vide, pour éviter les problèmes on ajoute un point 0 Rp, qui
        # n'aura pas d'impacte sur les barycentres muu
        while [] in S:
            S.remove([])
            S.append([np.zeros(p)])

        # on calcule le déplacement moyen des barycentres delta_muu pour la condition d'arrêt ainsi que les nouveaux
        # barycentres muu
        delta_muu = muu - barycentre(S)
        muu = barycentre(S)

    # --------------------------------------------- fin de l'algorithme ------------------------------------------------
    # on créer une matrice de taille (k, p + 1) pour simplifier l'utilisation des résultats
    #       [[S0, 0],
    #        [S1, 1], ...           avec Sk les éléments de S

    resultat = []
    for i, s in enumerate(S):
        b = np.block([[point, i] for point in s])
        resultat.append(b)

    # on retourne le partitionnement S trouver par l'algorithme des k-moyens
    return np.concatenate(resultat)


def load_image(file_name):
    img = cv2.imread(file_name)
    return img


def choix_points(img, nb_pixel, radius=1):
    # on choisi uniformement nb_pixel dans l'image exepter les bords de largeur radius
    h, l, _ = img.shape
    I = np.random.randint(radius, h - radius, size=nb_pixel)
    J = np.random.randint(radius, l - radius, size=nb_pixel)

    return I, J


def data_pixels(img, I, J, radius=1):
    # on créer une matrice data de taille (nb_pixel, 8) sur laquelle on fera une ACP
    data_img = np.zeros((len(I), 10))
    # on choisi 8 critères pour définir chaque pixel, on pourrait en choisir plus ou des différents en
    # fonction des application

    # pour chaque point choisi uniformement on enregistre dans data la pisition i j, la couleur r v b et la moyenne
    # des couleurs environnentes (carré de coté 2 * radius)
    for indice, (i, j) in enumerate(zip(I, J)):
        r, v, b = img[i, j]

        rm = img[i - radius:i + radius + 1, j - radius:j + radius + 1, 0].mean()
        vm = img[i - radius:i + radius + 1, j - radius:j + radius + 1, 1].mean()
        bm = img[i - radius:i + radius + 1, j - radius:j + radius + 1, 2].mean()

        y = (int(r) + int(v)) // 2
        ym = (rm + vm) / 2

        data_img[indice, :] = i, j, r, v, b, rm, vm, bm, y, ym

    # on retourn la matrice d'information
    return data_img


def ACP_img(data_img, w=None, analyse=None):
    # on fait une ACP pondéré classique sur la matrice data_img
    # les données sont normalisé mais ensuite pondérer par w

    n, d = data_img.shape
    Xcr = centre_red(data_img)

    if w is None:
        W = np.eye(d)
    else:
        W = np.diag(w)

    # on fait une svd sur la matrice de covariance C de taille (8, 8) donc le calcul ne depend
    # pas de nb_pixel
    C = (Xcr @ np.sqrt(W)).T @ (Xcr @ np.sqrt(W)) / np.sum(W)
    U, S, VT = np.linalg.svd(C)

    # on choisi de ne garder que les 2 premières composantes principales Y1, Y2 pour ensuite appliquer l'algo de Kmoyen
    # (c'est une consigne du sujet, l'algo de Kmean fonctionne pour n coordonnée pas uniquement 2)
    acp = Xcr @ U[:, :2]

    # pour definir de nombre de groupe dans l'image on utilise la règle de kaiser
    # ce qui conditionne le nombre de barycentre de l'algo des Kmoyen
    kaiser = 0
    while S[kaiser] > np.sum(S) / d:
        kaiser += 1

    if analyse is not None:
        mat_cor = correlationdirprinc(Xcr, 2, normalisation=False)
        cercle_cor(mat_cor, analyse, show=True)


    # on ne retourne pas uniquement le résultat de l'acp car on veut garder une trace des
    # coordonnées d'origine du pixel, chaque ligne de acp_indice est de la forme : i, j, y1, y2
    acp_indice = np.concatenate([data_img[:, :2], acp], axis=1)
    return acp_indice, kaiser


def Kmoy_img(data_img, kaiser, err=10 ** -15):
    # l'algo des k-moy image est en tout point identique au k-moy classique
    # la différence se fait sur la selection de donnée :
    #       data = [[...],
    #               [i, j, y1, y2],
    #               [...]]
    # on ne souhaite traiter dans l'algo que les y1, y2

    # ------------------------------------- initialistation de l'algorithme --------------------------------------------
    m, p = data_img.shape
    indice = np.random.choice(m, size=kaiser)
    while len(np.unique(indice)) != kaiser:
        indice = np.random.choice(m, size=kaiser)

    # on ne travaille que sur les données y1, y2 et non i, j
    muu = data_img[indice, 2:]
    delta_muu = muu
    S = None

    # ------------------------------------------- algorithme k-means ---------------------------------------------------
    while np.linalg.norm(delta_muu) > err:
        S = [[] for l in range(kaiser)]

        for i in range(m):
            # ai represente une ligne (donc un pixel) de data_img
            # on ne souhaite travailler que sur les y1 et y2, c'est a dire ai_val
            ai = data_img[i, :]
            ai_val = data_img[i, 2:]

            # on calcule les distances relatives aux barycentres a partir des ai_val
            D = [np.linalg.norm(ai_val - muu[j, :]) for j in range(kaiser)]
            j, _ = min(enumerate(D), key=lambda x: x[1])
            # on enregistre l'information des ai pour ne pas perdre les positions i, j des pixels en question
            S[j].append(ai)

        while [] in S:
            S.remove([])
            S.append([np.zeros(p)])

        # pour recalculer les barycentres, on utilise une fois de plus que les 2 dernières composantes y1, y2
        delta_muu = muu - barycentre(S)[:, 2:]
        muu = barycentre(S)[:, 2:]

    # --------------------------------------------- fin de l'algorithme ------------------------------------------------
    # on créer une matrice de taille (k, p + 1) pour simplifier l'utilisation des résultats
    #       [[S0, 0],
    #        [S1, 1], ...           avec Sk les éléments de S

    resultat = []
    for i, s in enumerate(S):
        b = np.block([[point, i] for point in s])
        resultat.append(b)

    # on retourne le partitionnement S trouver par l'algorithme des k-moyens
    return np.concatenate(resultat)


def masque(img, S, color):
    # la fonction masque retourne une matrice img_masque_ponctuel de la taille de img casiment vide (pixel noir)
    # les pixels classifier dans S par la fonction kmoy sont répartie sur l'image
    # pour chaque partion de S on associe une couleur
    h, l, _ = img.shape
    img_masque_ponctuel = np.zeros((h, l, 3))

    for points in S:
        i, j = points[:2]
        groupe = points[-1]
        img_masque_ponctuel[int(i), int(j)] = color[int(groupe)]

    # on convertit le type de valeur de la matrice en uint8 pour pour l'afficher en temps qu'image
    return img_masque_ponctuel.astype('uint8')


def remplissage_masque(img_masque_ponctuel):
    # le masque ponctuel est essentiellement composé de pixels noir, seulement quelque % sont des pixels calssifier
    # (pour classifier avec les kmoy l'ensemble des pixels de l'image cela demanderait trop de calcul)
    # le but est de faire diffuser les couleurs de l'image

    # on créer un masque en niveau de gris
    mask = cv2.cvtColor(img_masque_ponctuel, cv2.COLOR_BGR2GRAY)
    mask[mask > 0] = 255
    mask = 255 * np.ones(np.shape(mask)) - mask

    # la fonction inpaint réalise la diffusion des couleurs a partir de l'image ponctuel et du masque en gris
    # au final la matrice img_masque est une image ou 100% des pixels ont été  partitionner
    img_masque = cv2.inpaint(img_masque_ponctuel, np.uint8(mask), 3, cv2.INPAINT_NS)
    return img_masque


def true_color_masque(img_masque, img, kaiser):
    # une fois le masque de partitionnement de l'image terminer. On souhaite pour chaque partition créer une image
    # indépendante ou les pixels de couleur du masque sont remplacé par les vraies couleurs de l'image sur un font blanc
    # toutes ces sous images sont enregistré dans la liste mosaic
    h, l, _ = img.shape
    mosaic = []

    # pour chaque catégorie = nombre de kaiser choisie pour faire les k-moy
    for q in range(kaiser):
        sub_img = img.copy()

        # on créer un masque boolen qui séparer une couleur du reste de l'image
        if kaiser <= 3:
            mask_negatif = img_masque[:, :, q] > 127
        else:
            mask_negatif = (img_masque[:, :, q % 3] > 127) * (img_masque[:, :, (q + 1) % 3] > 127)

        # on fait le négatif du masque car on applique le masque sur l'image réel et non sur une image blanche
        mask = np.array(1 - mask_negatif, dtype=bool)
        # 255 correspond au blanc (on met 255 sur les 3 couches de l'image r v b)
        sub_img[mask] = 255

        mosaic.append(sub_img)
    return mosaic


def show(mosaic, img_masque, save=False):
    # la fonction show affiche et enregistre independement les sous image de la liste mosaic

    # file_name garantit un nom unique des fichiers
    file_name = time.asctime(time.localtime(time.time()))
    file_name = file_name.replace(' ', '_')
    file_name = file_name.replace(':', '-')

    for q, sub_img in enumerate(mosaic):
        cv2.imshow(str(q), sub_img)
        if save:
            cv2.imwrite(f'exo2_resultats/{file_name}_{str(q)}.jpg', sub_img)

    cv2.imshow(' ', img_masque)
    if save:
        cv2.imwrite(f'exo2_resultats/{file_name}.jpg', img_masque)

    cv2.waitKey()

