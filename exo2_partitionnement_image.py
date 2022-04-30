from Ma325.BE.exo2_lib import *



color = [(255, 0, 0),
         (0, 255, 0),
         (0, 0, 255),
         (255, 255, 0),
         (0, 255, 255),
         (255, 0, 255)]
img_path = ['feuille.jpg',      # 0
            'immeuble.jpg',     # 1
            'lena.png',         # 2
            'montagne.jpg',     # 3
            'neb.jpg',          # 4
            'paysage.jpg',      # 5
            'zebra1.jpg',       # 6
            'zebra2.jpg',       # 7
            'zebre.jpg',        # 8
            'batiment.jpg',     # 9
            'voiture.JPG',      # 10
            'piou.jpg']

# ---------------------------------------------- Paramètres ------------------------------------------------------------
nb_pixel = 5000
#    i  j  r  v  b  rm vm bm y ym
label = ['i', 'j', 'r', 'v', 'b', 'r.m', 'v.m', 'b.m', 'y', 'y.m']
w =     [ 1,   1,   1,   1,   1,   1,     1,     1,     0,   0]
zoom = 10
print("Partitionnement de l'image : ")
print(f"nb pixels = {nb_pixel}")
print("poids :  i  j  r  v  b  rm vm bm")
print("       ", w)

# -------------------------------------------- pré-traitement de l'image -----------------------------------------------
print("\t pré-traitement ", end=' ')
img = load_image('exo2_test/' + img_path[-1])
I, J = choix_points(img, nb_pixel, zoom)
data_img = data_pixels(img, I, J, zoom)
print("OK")


# -------------------------------------------- apprentissage non supervisé ---------------------------------------------
print("\t apprentissage non supervisé ", end=' ')
acp_img, kaiser = ACP_img(data_img, w, analyse=label)
categorie = 2
S = Kmoy_img(acp_img, categorie)
print("OK   ", kaiser)

# ----------------------------------------------- traitement de l'image ------------------------------------------------
print("\t traitement de l'image ", end=' ')
img_masque_ponctuel = masque(img, S, color)
img_masque = remplissage_masque(img_masque_ponctuel)
mosaic = true_color_masque(img_masque, img, categorie)
print("OK")

# ------------------------------------------------- affichage de l'image -----------------------------------------------
print("(appuyez sur q pour fermer les images)")
show(mosaic, img_masque, save=True)
