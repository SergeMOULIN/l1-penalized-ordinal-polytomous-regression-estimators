#-*-coding:utf8-*-

import numpy as np
from model_Polytomique_ordonne_v7 import ph, phprime, like, gradbet, quantif, lossFunction, draw1DGam
from model_Polytomique_ordonne_v7 import lonepoly
#from quasi_original import lonepoly
from random import *
import time
from numpy import *
import pylab
import matplotlib.pyplot as plt


#########################################################################################################
# I) Subroutines :
#########################################################################################################


def Normalisation(X):
    '''
    Normalize variables (i.e. normalize columns)
    '''
    for j in range(X.shape[1]):
        STD = np.std(X.take(j,axis = 1))
        M = np.mean(X.take(j,axis = 1))
        if STD != 0:
            for i in range (X.shape[0]):
                X[i,j] = (X[i,j]-M)/STD
    return(X)


def SujetsParCategorie(y, n, Q, m):
    res = np.zeros(Q)
    for i in range(n):
        for j in range(Q):
            if y[i] == m[j]:
                res[j] = res[j] + 1
    return res


def ponderation(y, n, Q, m):
    SujetsParCat = SujetsParCategorie(y, n, Q, m)
    res = np.zeros(Q)
    for i in range(Q):
        res[i] = 1. / SujetsParCat[i]
    return res


def bienClasse(X, y, bet, gam, m, Q):
    '''
     This fonction indicates how many subject are ranked correctly
     by the regression model.
    '''
    n = X.shape[0]
    yEtoileEstime = np.dot(X,bet)
    if bet == []:
        yEtoileEstime = np.zeros((n,1))
    cc=np.vstack([-float('inf'),gam,float('inf')])
    BienClasse = 0
    for i in range(n):
        ProbaMax = 0
        catObtimale = 0
        for j in range (Q):
            Proba = ph(cc[j+1,0] - yEtoileEstime[i,0]) - ph(cc[j,0] - yEtoileEstime[i,0])
            if Proba > ProbaMax:
                ProbaMax = Proba
                catObtimale = m[j]
        if catObtimale == y[i]:
            BienClasse += 1
    return BienClasse


def bienClassePondere(X, y, bet, gam, m, Q):
    '''
    '''
    n = X.shape[0]
    yEtoileEstime = np.dot(X,bet)
    if bet == []:
        yEtoileEstime = np.zeros((n,1))
    cc=np.vstack([-float('inf'),gam,float('inf')])
    BienClasse = 0
    Ponderation = ponderation(y = y, n = n, Q = Q, m = m)
    for i in range(n):
        ProbaMax = 0
        catObtimale = 0
        for j in range (Q):
            Proba = ph(cc[j+1,0] - yEtoileEstime[i,0]) - ph(cc[j,0] - yEtoileEstime[i,0])
            if Proba > ProbaMax:
                ProbaMax = Proba
                catObtimale = m[j]
                Pond = Ponderation[j]
        if catObtimale == y[i]:
            BienClasse += Pond
    return BienClasse



def errpredSurI(n,p,bet,gam,y,X,i,m):
    '''
    fonction qui indique l'erreur de prediction pour un sujet donnée.
    '''
    cc=np.vstack([-float('inf'),gam,float('inf')])
    Q = len(gam) + 1
    probs=np.zeros((Q,1))
    probmax=0
    imax=0
    for q in range(Q):
        probs[q,0]=ph(np.dot(X[i,:],bet)-cc[q,0])-ph(np.dot(X[i,:],bet)-cc[q+1,0])
        if probmax<probs[q,0]:
            probmax=probs[q,0]
            imax=m[q]
    res=abs(imax-y[i,0])
    return res

def errpred(n,p,bet,gam,y,X,m):
    res = 0
    for i in (range(n)):
        res += errpredSurI(n = n, p = p ,bet = bet ,gam = gam ,y = y ,X = X, i = i, m = m)
    return res


def makeInt(vecteur):
    '''
    fonction qui permet de transformer un vecteur d'entier considéré comme des réels en
    vecteur d'entiers considérés comme des entiers.
    Utilisé notament pour traiter le support qui ressort toujours comme un vecteur de réels.
    '''
    vecteur2 = []
    for k in range (len(vecteur)):
        vecteur2.append(int(vecteur[k]))
    return vecteur2


def CatMajoritaire(yApp, yTest, m):
    '''
    Retourne :
    -La categorie majoritaire de yApp.
    -Le nombre d'élement de yApp qui sont dans cette catégorie majoritaire.
    -Le nombre d'élement de yTest qui sont dans la catégorie majoritaire de yApp. . 
    '''
    nApp = yApp.shape[0]
    nTest = yTest.shape[0]
    catMajoritaire = 0
    tailleCatMajoritaire = 0
    for j in range(len(m)):
        cat = m[j]
        tailleCat = 0
        for i in range(nApp):
            if yApp[i] == cat:
                tailleCat += 1
        if tailleCat > tailleCatMajoritaire:
            catMajoritaire = cat
            tailleCatMajoritaire = tailleCat
    DansCatMajoritaire = 0
    for i in range(nTest):
        if yTest[i] == catMajoritaire:
           DansCatMajoritaire += 1
    return [catMajoritaire, tailleCatMajoritaire, DansCatMajoritaire]


def dicotomie(X, ySim, Q, m, L, method, lamb1, delta):
    '''
    Cette fonction est une sous-fonction a la fois de "quantile", de "classementBIC" et de "app_test_multiple".
    Elle permet, pour un X et y donné de retourner la valeur de lambda0 telle que tout les beta s'annulent.
    lamb1 : désigne la valeur de lambda dont on part pour démarer la dicotomie.
    delta : désigne la précision souhaitée.
    Tout les autres paramètres sont les parametres de lonepoly.
    '''
    lamb3 = lamb1
    # 1ere étape: Doubler lamb3 tant que tout les coefficients beta ne sont pas nul.
    while True:
        res = lonepoly(X, ySim, Q, m, nbIter = L, method = method, lamb = lamb3)
        if len(res[2]) == 0: # Si la longueur du support obtenu par lonepoly est nulle, on sort de la boucle.
            break
        #print "lamb3 = ", lamb3
        #print res[2]
        lamb3 = lamb3 * 2
    #print "lamb3 = ", lamb3
    #print res[2]
    lamb2 = lamb3 / 2.
    # 2eme étape: Rapprocher lamb2 et lamb3
    while (lamb3 - lamb2 > delta):
        lamb_moy = (lamb2 + lamb3)/2.
        res = lonepoly(X, ySim, Q, m, nbIter = L, method = method, lamb = lamb_moy)
        if len(res[2]) == 0:
            lamb3 = lamb_moy
        else :
            lamb2 = lamb_moy
    lamb = (lamb2 + lamb3)/2.
    #print "lamb dicotomie = ", lamb  
    return lamb


####################################################################################################
#II) Les 5 méthodes de choix de lambda
# Quantile_Norm_Inf, quantile, classementAIC, classementBIC et validationCroisee   
####################################################################################################


def norm_Inf(X,y,gam):
    '''
    Cette fonction permet de determiner (rapidement) une valeur de lambda0 telle que
    lambda0 = ||df(0)||inf. Cette valeur est donc assez élévée pour annuler beta.
    C'est une sous-fonction quantile_fast. 
    Avantage : C'est plus rapide que la fonction "dicotomie" car on a pas recour à lonepoly ici.
    Inconveniant : Ca implique de connaitre la valeur de gamma.
    '''
    n = X.shape[0]
    p = X.shape[1]
    bet = np.zeros((p,1))
    gbet = gradbet(X,y,n,p,bet,gam)
    Lamb = max(gbet)[0,0]
    return Lamb


def quantile_fast(X, y, gam, nbSim, alpha, seed):
    '''
    Cette fonction permet d'appliquer le quantile universal threshold en utilisant le lambda optenu par
    la fonction "norm_Inf" comme majorant. On considère ici que gamma est connu.
    C'est une sous fonction de la methode "Quantile_Norm_Inf"
    '''
    liste = []
    for l in range (nbSim):
        np.random.seed(seed * 300382 + l * 150913)
        ySim = np.random.permutation(y)
        liste.append(norm_Inf (X, ySim, gam))
    liste2 = sorted(liste)
    index = int(nbSim*(1-alpha))
    lamb0 = liste2[index]
    return lamb0


def Quantile_Norm_Inf(X, y, Q, m, L, method, nbSim, alpha, nbBoucle):
    '''
    Permet de determiner (relativement rapidement une valeur de lambda crédible.
    '''
    p = X.shape[1]
    lamb = sqrt(2) * log(2* max(p,1)) *  max(0.01,std(y))
    for i in range(nbBoucle):
        seed  = i
        gam = lonepoly(X, y, Q, m, nbIter = L, method = method, lamb = lamb)[1]
        lamb = quantile_fast(X, y, gam, nbSim, alpha, seed)
    return lamb
    

def quantile(X, y, Q, m, L, method, lamb1, delta, nbSim, alpha, seed):
    '''
    Cette fonction permet de choisir lambda par la méthode du  quantile universal threshold.
    nbSim : Nombre de fois qu'on simule un vecteur y et qu'on lui applique la fonction "dicotomie".
    alpha : Proportion des lambda obtenus supérieurs au lambda0 final.
    En d'autre termes, on prend lambda0 égal à la nbSim*(1-alpha) ieme position de la liste des lambda obtenus.
    '''
    liste = []
    for l in range (nbSim):
        np.random.seed(seed * 300382 + l * 150913)
        ySim = np.random.permutation(y)
        liste.append(dicotomie (X, ySim, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta))
    liste2 = sorted(liste)
    index = int(nbSim*(1-alpha))
    lamb0 = liste2[index]
    # tracer l'hystogramme.
    return lamb0


def classement_BIC_AIC(X, y, Q, m, L, method, lamb1, delta, L2, AIC_ou_BIC):
    '''
    Cette fonction permet de determiner le meilleur lambda à l'aide du BIC ou AIC.
    Dans un premier temps, on cherche un majorant à lambda en recherchant par dicotomie une valeur de
    lambda qui annule tout les beta.
    '''
    lambMax = dicotomie(X, y, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta)
    Min = float('inf')
    BestLamb = 0
    n = X.shape[0]
    p0 = X.shape[1]
    support1 = range(2*p0) # J'initialise support1 de sorte que ce ne soit pas un support possible.
    for i in range(L2):
        lamb = float(lambMax) * float(i) / (L2 - 1) 
        res = lonepoly(X, y, Q, m, nbIter = L, method = method, lamb = lamb) # Régression pénalisée.
        support2 = makeInt(res[2])
        if support2 != support1:
            Xs = X.take(support2,axis = 1)
            p = Xs.shape[1]
            res = lonepoly(Xs, y, Q, m, nbIter = L, method = method , lamb = 0) # Régression non pénalisée.
            bet = res[0]
            gam = res[1]
            logLikelihood = like(n, p, Xs, bet, gam, y)
            if AIC_ou_BIC == 'BIC':
                critere = p * log(n) - 2 * logLikelihood
            if AIC_ou_BIC == 'AIC':
                critere = 2 * p - 2 * logLikelihood    
            if critere < Min:
                Min = critere
                BestLamb = lamb
        support1 = support2
    return(BestLamb)
    

def validationCroisee(X, y, Q, m, L, method, lamb1, delta, L2):
    '''
    Cette fonction permet de détérminer le meilleur lambda à l'aide de la validation croisée. 
    '''
    n = X.shape[0]
    p = X.shape[1]
    lambMax = dicotomie(X = X, ySim = y, Q = Q, m = m, L = L, method = method, lamb1 = lamb1, delta = delta)
    VecteurBienClasse = np.zeros(L2) # Le vecteur qui conserve le nombre de sujets bien classés pour chaque valeur de lambda. 
    for i in range (n):
        XApp = X.take(indexApp,axis=0)
        indexApp = range(i)+range(i+1,n)
        XApp = X.take(indexApp,axis=0)
        yApp = y.take(indexApp,axis=0)
        support1 = range(2*p) # J'initialise support1 de sorte que ce ne soit pas un support possible.
        VecteurBienClasseProv = np.zeros(L2) 
        for k in range (L2):  # les différentes valeurs possibles de lambda.
            lamb = float(lambMax) * float(k) / (L2 - 1)
            res = lonepoly(XApp, yApp, Q, m, nbIter = L, lamb = lamb) # Régression pénalisée.
            support2 = makeInt(res[2])
            if support1 == support2:
                VecteurBienClasseProv[k] = VecteurBienClasseProv[k-1]
            if support1 != support2:
                Xs = X.take(support2,axis = 1)
                res = lonepoly(Xs, y, Q, m, nbIter = L, lamb = 0) # Régression non pénalisée.
                bet = res[0]
                gam = res[1]
                Xbet=np.dot(Xs,bet)
                yEtoileEstime = 0. + np.dot(Xs[i],bet)
                cc=np.vstack([-float('inf'),gam,float('inf')])
                ProbaMax = 0
                catObtimale = 0
                for j in range (Q): # Calcul de la proba d'être dans telle ou telle catégorie
                    Proba = ph(cc[j+1] - yEtoileEstime) - ph(cc[j] - yEtoileEstime)
                    if Proba > ProbaMax:
                        ProbaMax = Proba
                        catObtimale = m[j]
                if catObtimale == y[i]:
                    VecteurBienClasseProv[k] = 1
            support1 = support2
        VecteurBienClasse += VecteurBienClasseProv
    BestLamb = float(lambMax) * float(argmax(VecteurBienClasse)) / (L2 - 1)
    return(BestLamb)


#############################################################################
# III) Méthode pour le choix de r. 
############################################################################


def balayage0FW(X, y, Q, m, r1, iterMax, L, Quick = 'False'):
    n = X.shape[0]
    if (Quick == 'False'):
        res = lonepoly(X, y, Q, m, nbIter=n, method = 'OFW', r = r1, drawObj = False)
    if (Quick == 'True'):
        res = lonepoly(X, y, Q, m, nbIter=n, method = 'QOFW', r = r1, drawObj = False)
    h = res[4]
    hMax = h
    Best_r = r1
    Best_res = res
    GoAgain = True
    cpt = 0
    while (GoAgain and cpt < iterMax):
        r1 = r1 * 2
        if (Quick == 'False'):
            res = lonepoly(X, y, Q, m, nbIter=n, method = 'OFW', r = r1, drawObj = False)
        if (Quick == 'True'):
            res = lonepoly(X, y, Q, m, nbIter=n, method = 'QOFW', r = r1, drawObj = False)
        h = res[4]
        if h > hMax:
            hMax = h
            Best_r = r1
            Best_res = res        
        else:
            GoAgain = False
        cpt += 1
    A = np.matrix(range(10))
    A = np.log(4) * A / 9.
    A = np.exp(A)  
    A = A * Best_r / 2. # Une suite géométrique de 10 termes qui commence en Best_r /2 pour finir en Best_r *2.
    A = A.tolist()[0]
    for r2 in A:
        if (Quick == 'False'):
            res = lonepoly(X, y, Q, m, nbIter= L, method = 'FW', r = r1, drawObj = False)
        ##################  à vérifier.
        h = res[4]
        if h > hMax:
            hMax = h
            Best_r = r1
            Best_res = res
    return Best_res, Best_r   


#################################################################################################
# IV) Génération d'une base de données 
##################################################################################################



def SimuleData (n, p, Q, m, s, betmag, Cgam, mc):  
    '''
    #s : Nombre de variables influentes. La valeur de s influe sur la simulation de bet0.
    #Cgam : Coéfficients multiplicateur permettant de créer gam0.
    #betmag : Coefficient multiplicateur qui indique à quel point les coeffients de bet0 sont importants. La valeur de s influe sur la simulation de bet0.
    '''

    # parameter generation
    #gam0 = Cgam * np.vstack([-3,-1,1,3])
    gam0 = Cgam * np.vstack([-0, 3])
    np.random.seed((mc +1) * 3071991)
    bet0=np.vstack([betmag*np.random.randn(s,1), 0.5*np.zeros((p-s,1))])
    # np.random.randn(s,1) = vecteur de taille (s,1) de valeurs suivants une loi normale.
    # Du coup, on a s éléments qui suivent un loi N(0,betmag)
    # Le reste des composantes du vecteur sont nulles.
    np.random.shuffle(bet0)

    # data generation
    np.random.seed((mc + 1) * 15092013)
    X = np.sqrt(n)**(-1) * np.random.randn(n,p)
    X = Normalisation(X)
    np.random.seed((mc + 1) * 30031982)
    u = np.random.random((n,1))
    np.random.seed((mc + 1) * 300555)
    nu0 = np.asmatrix(np.dot(X,bet0))
    eps = np.asmatrix(log(u *((1 - u)**(-1)))) # Le bruit de y* 
    y =quantif(n, Q, nu0 + eps, gam0, m)
    '''
    # Histogrammes (facultatifs)
    
    res = plt.hist(nu0, bins = 15)
    plt.title('nu0')
    plt.show()
    res = plt.hist(eps, bins = 15)
    plt.title('eps')
    plt.show()
    res = plt.hist(nu0+eps, bins = 15)
    plt.title('nu + eps')
    plt.show()
    res = plt.hist(y) #, bins = 15)
    plt.title('y')
    plt.show()
    
    print "nu0+eps = ", nu0+eps
    print "y = ", y 
    '''
    '''
    for i in range(Q-1):
        draw1DGam(n, p, XApp, np.zeros((p,1)), np.vstack(range(Q-1))- Q/2. + 1, m, yApp, 0.1, 0.005, i, 100)
    '''
    return (X, y)

    

################################################################################################
# V) Tests et comparaisons de nos méthodes de choix de lambda
################################################################################################


def apprentissage_test(X, y, Q, m, L, method, lamb1, delta, L2, alpha, function, seed):
    '''
    Cette fonction sépare l’échantillon total en échantillon d'apprentissage et
    échantillon de test. Elle permet d'estimer les paramètres (lambda, Beta, Gamma) sur l’échantillon d'apprentissage
    pour ensuite voir si ces parmètres permetent une estimation correcte sur l’échantillon de test.
    C'est une sous-fonction à la fois de "app_test_multiple" et de "MonteCarlo3" 
    fonction : Chaîne de caractère qui indique si l'evaluation de lambda ce fait via quantile universal threshold ou via BIC
    ou via validation croisée.
    '''
    n = y.shape[0]
    nApp = (n*2) / 3
    nTest = n - nApp # Calcul de la taille des échantillons d'apprentissage et de test.

    np.random.seed(seed * 20061984)
    index0 = range(n)
    index = np.random.permutation(index0)
    indexApp = index[:nApp]
    indexTest = index[nApp:] # Tirage de l'index des éléments des echantillons d'apprentissage et de test.  
    XApp = X.take(indexApp,axis=0)
    XTest = X.take(indexTest,axis=0)
    yApp = y.take(indexApp,axis = 0) 
    yTest = y.take(indexTest,axis=0) # Constructions des echantillons d'apprentissage et de test.

    m = np.unique(y.tolist())
    Q = len(m)
    time1 = time.time()
    if function == 'quantile':
        lamb0 = quantile(XApp, yApp, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, nbSim = L2, alpha = alpha, seed = seed)
    if function == 'BIC':
        lamb0 = classement_BIC_AIC(XApp, yApp, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, L2 = L2, AIC_ou_BIC = 'BIC')
    if function == 'AIC':
        lamb0 = classement_BIC_AIC(XApp, yApp, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, L2 = L2, AIC_ou_BIC = 'AIC')
    if function == 'validationCroisee':
        lamb0 = validationCroisee(XApp, yApp, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, L2 = L2)
    if function == 'Quantile_Norm_Inf':
        nbBoucle = 3
        lamb0 = Quantile_Norm_Inf (XApp, yApp, Q, m, L = L, method = method, nbSim = L2, alpha = alpha, nbBoucle = nbBoucle)
    if function == 'zero':
        lamb0 = 0


    if (function in ['quantile', 'BIC', 'AIC', 'validationCroisee', 'Quantile_Norm_Inf', 'zero']):   
        res = lonepoly(XApp, yApp, Q, m, L, method = method, lamb = lamb0, frein = True,
                         drawObj = False, drawPartial = False)
        support = makeInt(res[2])
        XsApp = XApp.take(support,axis = 1)
        res = lonepoly(XsApp, yApp, Q, m, L, method = method, lamb = 0, frein = True,
                         drawObj = False, drawPartial = False) # Régression non pénalisée. 
        lamb0_Ou_r = lamb0

    if function == 'OFW':
        res0 = balayage0FW(XApp, yApp, Q, m, 0.1, 12, L = L)
        #res = res0[0]
        lamb0_Ou_r = res0[1]
        res = lonepoly(XApp, yApp, Q, m, nbIter = L, method = 'FW', r = lamb0_Ou_r , drawObj = False)
        support = makeInt(res[2])

    if function == 'QOFW':
        res0 = 0  ##### A modifier


    DansCatMajoritaire = CatMajoritaire(yApp, yTest, m)[2]
    
    if function == 'modeleNul':
        return (DansCatMajoritaire, 0, 0, 0, DansCatMajoritaire, 0, 0, 0)

    bet = res[0]
    gam = res[1]
    p = len(support)
    time2 = time.time()
    difTime = time2 - time1

    XTest = XTest.take(support, axis=1)
    BienClasse = bienClasse(XTest, yTest, bet, gam, m, Q)
    LogLike = like(n = nTest, p = p, X = XTest, bet = bet,  gam = gam, y = yTest)
    Errpred = errpred(n = nTest, p = p, bet = bet, gam = gam, y = yTest, X = XTest, m = m)
    BienClassePond = bienClassePondere(XTest, yTest, bet, gam, m, Q)
    #print "function = ", function, "method = ", method, "time = ", difTime  
   
    return (BienClasse, LogLike, Errpred, BienClassePond, DansCatMajoritaire, difTime, lamb0_Ou_r, len(support))


def app_test_multiple(X, y, Q, m, L, method, lamb1, delta, L2, alpha, L3, function, SeedStart):
    '''
    Cette fonction permet de considérer un grand nombre d'echentillons d'apprentissage différents 
    L3 est nombre d'echantillons d'apprentissage considéré.
    Tout les autre paramètres sont les paramètres de la function "apprentissage_test".
    '''
    BienClasse = 0
    LogLike = 0
    Errpred = 0
    BienClassePond = 0
    DansCatMajoritaire = 0
    Time = 0
    lamb0_Ou_r = 0
    nbGenes = 0
    distribBC = []
    distribLike = []
    distribErr = []
    distribBCP = []
    
    for l in range(SeedStart, L3 + SeedStart):
        print "l = ", l
        seed = l
        res = apprentissage_test(X, y, Q, m, L = L, method = method, lamb1 = lamb1,
                                  delta = delta, L2 = L2, alpha = alpha, function = function, seed = seed)
        BienClasse += res[0]
        LogLike += res[1]
        Errpred += res[2]
        BienClassePond += res[3]
        DansCatMajoritaire += res[4]
        Time += res[5]
        lamb0_Ou_r += res[6]
        nbGenes += res[7]
        distribBC.append(res[0])
        distribLike.append(float(res[1]))
        distribErr.append(float(res[2]))
        distribBCP.append(res[3])

    # L3 = 10
    # n =  1656
    
    n = y.shape[0]
    nTest = n - ((n*2) / 3)
    percentage_succes = 100. * float(BienClasse) / (nTest * L3)
    LogLike = float(LogLike) / (nTest * L3)
    Like = np.exp(LogLike)
    meanErrpred = float(Errpred) / (nTest * L3)
    meanBienClassePond = 100. * float(BienClassePond) / (Q * L3)
    percentage_catMajoritaire = 100. * float(DansCatMajoritaire) / (nTest * L3)   
    meanTime = float(Time) / L3
    meanLamb = float(lamb0_Ou_r) / L3
    meanNbGenes = float(nbGenes) / L3

    return (percentage_succes, Like, meanErrpred, meanBienClassePond, percentage_catMajoritaire,
            meanTime, meanLamb, meanNbGenes, distribBC, distribLike, distribErr, distribBCP) 


def MonteCarlo3(n, p, Q, m, s, betmag, Cgam, MC, L, method, lamb1, delta, L2, alpha, function):
    '''
    '''
    BienClasse = 0
    LogLike = 0
    Errpred = 0
    BienClassePond = 0
    DansCatMajoritaire = 0
    Time = 0
    lamb0_Ou_r = 0
    nbGenes = 0
    distribBC = []
    distribLike = []
    distribErr = []
    distribBCP = []
    
    for mc in range(MC):
        print "mc = ", mc
        X, y  = SimuleData (n = n, p = p, Q = Q, m = m, s = s, betmag = betmag, Cgam = Cgam, mc = mc) 
        res = apprentissage_test(X = X, y = y, Q = Q, m = m, L = L, method = method, lamb1 = lamb1,
                           delta = delta, L2 = L2, alpha = alpha, function = function, seed = mc)
        BienClasse += res[0]
        LogLike += res[1]
        Errpred += res[2]
        BienClassePond += res[3]
        DansCatMajoritaire += res[4]
        Time += res[5]
        lamb0_Ou_r += res[6]
        nbGenes += res[7]
        distribBC.append(res[0])
        distribLike.append(float(res[1]))
        distribErr.append(float(res[2]))
        distribBCP.append(res[3])

    nTest = n - ((n*2) / 3)    
    percentage_succes = 100. * float(BienClasse) / (nTest * MC)
    LogLike = float(LogLike) / (nTest * MC)
    Like = np.exp(LogLike)
    meanErrpred = float(Errpred) / (nTest * MC)
    meanBienClassePond = 100. * float(BienClassePond) / (Q * MC)
    percentage_catMajoritaire = 100. * float(DansCatMajoritaire) / (nTest * MC)   
    meanTime = float(Time) / MC
    meanLamb = float(lamb0_Ou_r) / MC
    meanNbGenes = float(nbGenes) / MC
    
    return (percentage_succes, Like, meanErrpred, meanBienClassePond, percentage_catMajoritaire,
            meanTime, meanLamb, meanNbGenes, distribBC, distribLike, distribErr, distribBCP) 



def Main(X, y, function = 'Quantile_Norm_Inf', L = 200, method = 'N', lamb1 = 1, delta = 0.01, alpha = 0.05, L2 = 100, nbBoucle = 3, HaveToBeNormalized = 'True'):
    '''
    Mandatory inputs:
        X : The matrix of explanatory variables.
        y : The variable to explain.
    Optional inputs: 
        function : The way to choose lambda (or r).
        L : The number of iterations of the optimization algorithm (i.e. Nesterov, or Gradient or FW according to the case).
        method : The optimization algorithm. Can be Nesterov ('N') or Gradient ('GS') or FW ('FW').
        lamb1 : The starting value used in the "dicotomie" subroutine.
        delta : The desired accuracy used in the "dicotomie" subroutine.
        alpha : The risk that Beta is not zero when y is not related to X. This parameter is used in "quantile_fast" which is a subroutine of the "Quantile_Norm_Inf" function.
        L2 : This parameter has different meanings depending on the value of the "fonnction" parameter. For example, if function = "Quantile_Norm_Inf", L2 is the number of simulated databases.
        nbBoucle : The number of loops in "Quantile_Norm_Inf" function.
        HaveToBeNormalized : Booleen argument that indicates whether X should be normalized. The user can set this argument to 'False' to save time if X is already normalized. 
    '''
    X = Normalisation(X)
    #print "y = ", y
    #print "y.shape = ", y.shape
    m = np.unique(y.tolist())
    #print "y = ", y
    Q = len(m)
    n = len(y)
    p = X.shape[1]
    
    if function == 'BIC':
        lamb0 = classement_BIC_AIC(X, y, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, L2 = L2, AIC_ou_BIC = 'BIC')
    if function == 'AIC':
        lamb0 = classement_BIC_AIC(X, y, Q, m, L = L, method = method, lamb1 = lamb1, delta = delta, L2 = L2, AIC_ou_BIC = 'AIC')
    if function == 'Quantile_Norm_Inf':
        lamb0 = Quantile_Norm_Inf (X, y, Q, m, L = L, method = method, nbSim = L2, alpha = alpha, nbBoucle = nbBoucle)
    if function == 'zero':
        lamb0 = 0
        
    if (function in ['BIC', 'AIC', 'Quantile_Norm_Inf', 'zero']):   
        res = lonepoly(X, y, Q, m, L, method = method, lamb = lamb0, frein = True, drawObj = False, drawPartial = False)
        support = makeInt(res[2])
        Xs = X.take(support,axis = 1)
        res = lonepoly(Xs, y, Q, m, L, method = method, lamb = 0, frein = True, drawObj = False, drawPartial = False) # Régression non pénalisée. 
        
    if function == 'OFW':
        res0 = balayage0FW(XApp, yApp, Q, m, 0.1, 12, L = L)
        r = res0[1]
        res = lonepoly(XApp, yApp, Q, m, nbIter = L, method = 'FW', r = r , drawObj = False)
        support = makeInt(res[2])

    bet = res[0]
    gam = res[1]

    return (support, bet, gam)

#X = load('newX.npy')
y = load('newY.npy')
#res = Main(X, y)
#print res 
