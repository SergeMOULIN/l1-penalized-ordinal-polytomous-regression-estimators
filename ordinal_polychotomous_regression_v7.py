# -*- coding: utf-8 -*-
# By Stephane Chretien (Laboratoire de Mathematiques, Universite de Franche Comte, 16 route de Gray, F-25000 Besancon, France. stephane.chretien@univ-fcomte.fr)


#import cvxopt as C
#from cvxopt import solvers
import numpy as np
from numpy import *  
from pylab import load, save
from time import time
import scipy as Sci
import scipy.linalg
import matplotlib.pyplot as plt
#import pickle
from scipy.optimize import minimize 
#read : https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html


######################################################################################################################
# I) Subroutines of lonepoly
######################################################################################################################


def quantif(n,Q,nu,gam,m):
    '''
    Cette fonction indique à partir d'une valeur de Y* dans quelle catégorie le patient est classé.
    nu : Y*, c'est un vecteur de taille n.
    n = nombre de sujets.
    m : Ensemble des catégories possibles.
    Q = nombre de catégories possibles.
    gam = Le vecteur des cut off entre catégories.
    '''
    y = np.zeros((n,1))
    for i in range(n):
        if nu[i,0] < gam[0,0]:
            y[i,0] = m[0,0]
        if nu[i,0] > gam[Q-2,0]:
            y[i,0] = m[Q-1,0]
        for q in range(1,Q-1):
            if gam[q-1,0] < nu[i,0] and nu[i,0] <= gam[q,0]:
                y[i,0] = m[q,0]
    return y


def ph(z):
    '''
    La fonction logistique. Cette fonction peut s appliquer à un vecteur comme à un scalaire.
    Remarque : Cette fonction peut provoquer des warnings (exponentielle d'un nombre trop grand) mais
    au final on obtient bien le nombre voulu (si z est grand, ph(z) = 1).
    '''
    #z = np.asmatrix(z) # Ne résout pas le soucis.
    #z=np.array(z,dtype=np.float32) # Crée un soucis à un autre endrois
    z = np.asmatrix(z,dtype=np.float32) # Ce n'est pas très satifaisant de perdre du temps de calcul à faire cela mais ça évite des erreurs
    valph = 1 - 1. / (1 + exp(z))
    return valph


def phprime(z):
    '''
    La dérivée de la fonction logistique.
    Cette fonction peut s appliquer à un vecteur comme à un scalaire.
    Remarque : Cette fonction peut provoquer des warnings (exponentielle d'un nombre trop grand) mais
    au final on obtient bien le nombre voulu (si z est grand, phprime(z) = 1).
    '''
    #z = np.asmatrix(z)
    #z=np.array(z,dtype=np.float32)
    z = np.asmatrix(z,dtype=np.float32)
    valphprime = (1 - 1. / (1. + exp(z))) / (1. + exp(z))
    return valphprime


def gradbet(X,y,n,p,bet,gam):
    '''
    La derivé de la log vraissemblance par rapport à beta.
    gam : vecteur des cut off. Vecteur vertical de taille Q -1.
    cc : gamma (les cut off)
    tutu : différence entre Ŷ* et le gamma inférieur (Ŷ* - gamma_{q-1}).
    titi : différence entre Ŷ* et le gamma supérieur (gamma_q - Ŷ*).
    retun gbet : Un vecteur de taille p.
    '''
    gam=np.asmatrix(gam)
    bet=np.asmatrix(bet)
    cc=np.vstack([-float('inf'),gam,float('inf')])
    tutu=np.dot(X,bet) # numphy.dot sert à calculer un produit scalaire. bet est un vecteur vertical de taille p. tutu est vertical. titi aussi.
    titi=np.dot(X,bet) # Titi = Ŷ* ici. C'est un vecteur colonne de taille n.
    tutu = np.matrix(tutu,dtype=np.float32)
    titi = np.matrix(titi,dtype=np.float32)


    for i in range(n):
        tutu[i,0] = tutu[i,0] - cc[int(y[i,0])-1,0]   # int c'est la partie entière. Ca permet d'éviter les bug si les états de gravité des patients sont été entrés comme des nombre réels. cc[int(y[i,0])] c'est la valeur de gamma_q-1 
        titi[i,0] = titi[i,0] - cc[int(y[i,0]),0]
    # tutu est toujours plus grand que titi.

    Indicatrice = np.array((ph(tutu)-ph(titi) < 1e-10))
    Numerateur = np.array(phprime(tutu) - phprime(titi)) * (1-Indicatrice)
    Denominateur = maximum(ph(tutu)-ph(titi), Indicatrice)
    toto = (Numerateur / Denominateur) - Indicatrice
    #toto=phprime(tutu)-phprime(titi)/maximum(1e-7,ph(tutu)-ph(titi)) # Le code avec l'erreur
    ##### C'est ici qu'il y a une modification à faire pour voir si vraimment les resultats sur données réelles sont meilleure avec l'erreur (oublie de parenthèse) que maintant que l'ereur est corrigée. (ce qui est bizzarre). 

    # L'objectif de maximum (1e-10 ...) est d'éviter les division  0/0.
    # Si ph(tutu)-ph(titi) est proche de 0 alors tutu est proche de titi alors phprime(tutu) - phprime(titi) est proche de 0.     
    # On ajoute - (ph(tutu)-ph(titi) < 1e-10) parce que toto tend vers -1 quand tutu et titi sont proche pas vers 0. 
    gbet=np.dot(np.transpose(X),toto)  # gbet est un vecteur colonne de taille p car matrice p*n multiplié par colonne vecteur de taille n.
    return gbet


def gradbetPenalise(X,y,n,p,bet,gam,lamb,mu):
    '''
    Cette fonction fournis le gradian en beta de la log vraissemblance pénalisé.
    '''
    gbet=gradbet(X,y,n,p,bet,gam)
    gradlonemu=np.zeros((p,1))
    for j in range(p):
        if abs(bet[j,0]) < mu:
            gradlonemu[j,0] = float(bet[j,0]) / mu
        else:
            gradlonemu[j,0] = float(bet[j,0]) / abs(bet[j,0])
    return gbet - lamb * gradlonemu


def gradgam(X, y, n, p, m, bet, gam):
    '''
    La dérivé de la log vraissemblance par rapport à gamma.
    tutu : difference entre Y* et le gamma inferieur (Ŷ* - gamma_{q-1}).
    titi : difference entre Y* et le gamma supérieur (gamma_q - Ŷ*).
    '''
    gam=np.asmatrix(gam)  # "vecteur colonne" Q-1
    bet=np.asmatrix(bet)  # vecteur colonne p
    cc=np.vstack([-float('inf'),gam,float('inf')])  # Vecteur colonne. De taille Q + 1.
    ggam=np.zeros((len(gam),1)) 
    for q in range(len(gam)):
        tata = np.dot(X,bet) - cc[q,0] # Vecteur colonne de taille n.
        tutu = np.dot(X,bet) - cc[q+1,0]
        titi = np.dot(X,bet) - cc[q+2,0]
        tempa = phprime(tutu) / maximum(1e-7,ph(tata)-ph(tutu)) 
        tempaa = tempa[(y==m[q])]
        tempb = phprime(tutu)/maximum(1e-7,ph(tutu)-ph(titi))
        tempbb = tempb[(y==m[q+1])]
        ggam[q] = tempaa.sum(axis=1) - tempbb.sum(axis=1)
    return ggam


def Iteration_Nesterov(X,y,thet,bet,gam,thet0,mu,l,p,Q,n,m,lamb,G,frein):
    '''
    Cette sous fonction produit une itération de l'algoritme de Nesterov.
    Notons que si on dé-commente la ligne "thetBis = ouaille" on obtient un algorithme du gradian simple
    (on peut alors commenter les 4 ligne precedenté).
    '''
    #print "CoinCoin"
    mu0 = mu
    gbetPena=gradbetPenalise(X,y,n,p,bet,gam, lamb, mu)
    ggam=gradgam(X,y,n,p,m,bet,gam) 
    g = .1 * np.vstack([-gbetPena, -ggam])
    while True:
        GBis = np.hstack([g,G])  # Est ce que g doit être pondéré par la division qu'on a fait subir à mu ici???? 
        ouaille = thet - mu * g     # B^(k,1) dans l'article
        alph = (np.arange(l+1) + 1) # arange(x) fait la même chose que range (x) en fait sauf que ça crée un array plutôt qu'une liste.
        alph = 0.5 / alph
        zed = thet0 - mu * np.dot(GBis,alph) # B^(k,2) dans l'article.
        thetBis = (2. / (l+4)) * zed + (1 - 2. / (l + 4)) * ouaille
        betBis = thetBis[range(0,p),0]
        gamBis = thetBis[range(p,p+Q-1),0]
        for q in range(Q-2):
            if gamBis[q+1] - gamBis[q] < 10**(-8):
                gamBis[q+1] =  gamBis[q] + 10**(-8)
        if ((not frein) or (lossFunction(n, p, X, betBis, gamBis, y, lamb, mu0) > lossFunction(n, p, X, bet, gam, y, lamb, mu0)) or (mu < 10**(-16))):
            break
        mu = mu / 2.
        #print mu
    return (thetBis, betBis, gamBis, GBis)


def IterationGradianSimple(X,y,thet,bet,gam,mu,p,Q,n,m,lamb,frein):
    '''
    '''
    mu0 = mu
    gbetPena=gradbetPenalise(X,y,n,p,bet,gam, lamb, mu)
    ggam=gradgam(X,y,n,p,m,bet,gam) 
    g = .1 * np.vstack([-gbetPena, -ggam])
    while True:
        thetBis = thet - mu * g    
        betBis = thetBis[range(0,p),0]
        gamBis = thetBis[range(p,p+Q-1),0]
        for q in range(Q-2):
            if gamBis[q+1] - gamBis[q] < 10**(-8):
                gamBis[q+1] =  gamBis[q] + 10**(-8)
        if ((not frein) or (lossFunction(n, p, X, betBis, gamBis, y, lamb, mu0) > lossFunction(n, p, X, bet, gam, y, lamb, mu0)) or (mu < 10**(-16))):
            break
        mu = mu / 2.
    return (thetBis, betBis, gamBis)


def IterationFW(X,y,thet,bet,gam,p,Q,n,m,r,l):
    '''
    Dans cette version j'ai fait un "frein" pour m'assurer que le pas en gamma ne soit pas trop grand.
    '''
    gbet = gradbet(X,y,n,p,bet,gam)
    C = (float((2.)) / float(l + 2))
    i = argmax(abs(gbet))
    if (gbet[i] != 0):
        Signe = gbet[i] / abs(gbet[i])
    else :
        Signe = 0
    bet = bet * (1 - C)
    bet[i] = bet[i] + Signe * r * C
    mu = 0.1
    ggam = gradgam(X,y,n,p,m,bet,gam)
    gamBis = np.copy(gam)
    i = argmax(abs(ggam))
    if (ggam[i,0] != 0):
        Signe = ggam[i,0] / abs(ggam[i,0])
    else :
        Signe = 0 
    while(True):
        MuGgam = mu * ggam
        if ((i <= Q - 3) and (Signe == 1)):
            gamBis[i,0] = min((gam[i,0] + MuGgam[i,0]), (gam[i+1,0] + gam[i,0])/2)
        else :
            if ((i >= 1) and (Signe == -1)):
                gamBis[i,0] = max((gam[i,0] + MuGgam[i,0]), (gam[i,0] + gam[i-1,0])/2)
            else :
                gamBis[i,0] = gam[i,0] + MuGgam[i,0]
        if (like(n,p,X,bet,gamBis,y) >= like(n,p,X,bet,gam,y)) or mu < 0.01:
            break
        mu = mu/2
        #print mu
    #print "gamBis = ", gamBis
    #print
    #print
    thet = np.vstack([bet, gamBis])
    return (thet, bet, gamBis)


def IterationOFW(X_l,y_l,thet,bet,gam,p,Q,n,m,r,l):
    gbet = gradbet(X_l,y_l,n,p,bet,gam)
    C = (float((2.)) / float(l + 2))
    i = argmax(abs(gbet))
    if (gbet[i] != 0):
        Signe = gbet[i,0] / abs(gbet[i,0])
    else :
        Signe = 0
    bet = bet * (1 - C)
    bet[i,0] = bet[i,0] + Signe * r * C
    mu = 0.1
    ggam = mu * gradgam(X_l,y_l,n,p,m,bet,gam)
    gamBis = np.copy(gam)
    #print "l = ", l
    while(True):
        MuGgam = mu * ggam
        for i in range(Q-1):                
            if (ggam[i,0] != 0):
                Signe = ggam[i,0] / abs(ggam[i,0])
            else :
                Signe = 0 
            if ((i <= Q - 3) and (Signe == 1)):
                gamBis[i,0] = min((gam[i,0] + MuGgam[i,0]), (gam[i+1,0] + gam[i,0])/2)
            else :
                if ((i >= 1) and (Signe == -1)):
                    gamBis[i,0] = max((gam[i,0] + MuGgam[i,0]), (gam[i,0] + gam[i-1,0])/2)
                else :
                    gamBis[i,0] = gam[i,0] + MuGgam[i,0]
        if like(n,p,X_l,bet,gamBis,y_l) >= like(n,p,X_l,bet,gam,y_l):
            break
        mu = mu/2
    thet = np.vstack([bet, gamBis])
    return (thet, bet, gamBis)

'''
def IterationQOFW(X_l,y_l,thet,bet,gam,p,Q,n,m,r,l):
    # A modifier...
    print "l  ", l
    gbet = gradbet(X_l,y_l,n,p,bet,gam)
    C = (float((2.)) / float(l + 2))
    i = argmax(abs(gbet))
    if (gbet[i] != 0):
        Signe = gbet[i,0] / abs(gbet[i,0])
    else :
        Signe = 0
    bet = bet * (1 - C)
    bet[i,0] = bet[i,0] + Signe * r * C
    mu = 0.1
    ggam = mu * gradgam(X_l,y_l,n,p,m,bet,gam)
    gamBis = np.copy(gam)
    #print "l = ", l
    while(True):
        MuGgam = mu * ggam
        for i in range(Q-1):                
            if (ggam[i,0] != 0):
                Signe = ggam[i,0] / abs(ggam[i,0])
            else :
                Signe = 0 
            if ((i <= Q - 3) and (Signe == 1)):
                gamBis[i,0] = min((gam[i,0] + MuGgam[i,0]), (gam[i+1,0] + gam[i,0])/2)
            else :
                if ((i >= 1) and (Signe == -1)):
                    gamBis[i,0] = max((gam[i,0] + MuGgam[i,0]), (gam[i,0] + gam[i-1,0])/2)
                else :
                    gamBis[i,0] = gam[i,0] + MuGgam[i,0]
        if like(n,p,X_l,bet,gamBis,y_l) >= like(n,p,X_l,bet,gam,y_l):
            break
        mu = mu/2
    thet = np.vstack([bet, gamBis])
    return (thet, bet, gamBis)
'''



#################################################################################################################################################
# II) Main function : lonepoly  
#################################################################################################################################################


def lonepoly(X, y, Q, m, nbIter, method, lamb = 0, r = 1, nbRestart = 1, frein = True, drawObj = False, drawPartial = False, thbet = 10 , thgam = 3):
    '''
    fonction qui résout le modèle polytomique ordonné sparse à lambda (ou r) fixé.
    5 argument obligatoires 5 argument optionnels de methode, 2 arguments de dessin dessins, 2 inutiles maintenant (grade au cas ou).
    Remarque : On peut également appliquer des "restart" si necessaire (d'ou la presence de deux boucles for imbriquées avec l1 et l2) 
    X : la Matrice patients * genes.
    y : Le vecteur des détériorations.
    n : Nombre de patients. 
    p : Nombre de gènes.
    Q : Le nombre d'états de détérioration.
    lamb : La pénalisation.
    L : Le nombre d'itérations. Fixé à l'avance. (noté N dans l'article). 
    thbet et thgam :  Limites à ne pas franchir pour les valeur de beta et gama (on peut les fixer à +- inf finalement.
    Support = Vecteur de taille p qui vaut 0 pour les gènes non influants et 1 pour les gènes influants.
    supp = Vecteur de taille inférieur à p qui garde l'indexe des gènes influants.
    method possibles = Nesterov (N) GradianSimple (GS) Mixte (M) Frank-Wolf (FW) Online Frank-Wolfe (OFW)  
    '''
    # Conditions initiales
    n = X.shape[0]
    p = X.shape[1]
    bet0 = np.zeros((p,1))
    gam0 = np.vstack(range(Q-1)) - Q/2. + 1  
    bet = bet0
    gam = gam0
    thet0 = np.vstack([bet,gam])  # Serge : vstack merœge verticalement les données tout simplement.
    thet=np.vstack([bet,gam])    
    mu = 0.005
    h = 0 # indicateur de la qualité de l'évaluation pour un r donné (OFW).  
    if (method == 'OFW'):
        X_l = np.zeros((0,p))
        y_l = np.zeros((0,1))
    if drawObj:
        GraphlossFunction = [] # Permet de tracer le graphique de l'évolution de la fonction objective en fonction des itérations.
        lossFunction = lossFunction(n, p, X, bet, gam, y, lamb, mu)
        GraphlossFunction.append(lossFunction)

    # Optimisation de la fonction objective.
    for l1 in range (nbRestart):
        if (method == 'N' or method == 'M'):
            G = np.zeros((p+Q-1,0)) # G est une matrice à p+Q-1 ligne et 0 colonnes. 
        for l2 in range(nbIter):
            if (method == 'GS'):
                res = IterationGradianSimple(X,y,thet,bet,gam,mu,p,Q,n,m,lamb,frein)
            if (method == 'N'):
                res = Iteration_Nesterov(X,y,thet,bet,gam,thet0,mu,l2,p,Q,n,m,lamb,G,frein)
                G = res[3]
            if (method == 'M' and l2 < nbIter/5):
                res = IterationGradianSimple(X,y,thet,bet,gam,mu,p,Q,n,m,lamb,frein)
            if (method == 'M' and l2 >= nbIter/5):
                res = Iteration_Nesterov(X,y,thet,bet,gam,thet0,mu,l2-nbIter/5,p,Q,n,m,lamb,G,frein)
                G = res[3]
            if (method == 'FW'):
                res = IterationFW(X,y,thet,bet,gam,p,Q,n,m,r,l2)
            if (method == 'OFW'):
                h = h + like(n,p,X,bet,gam,y)
                if (l2%n == 0):
                    np.random.seed(2111983 + l2/n)
                    permu = np.random.permutation(range(n))
                index = permu[l2%n]
                w_l_X = np.matrix(X.take(index, axis = 0)) 
                w_l_y = np.matrix(y.take(index, axis = 0))
                X_l = np.concatenate((X_l, w_l_X), axis = 0)
                y_l = np.concatenate((y_l, w_l_y), axis = 0)
                res = IterationOFW(X_l,y_l,thet,bet,gam,p,Q,l2+1,m,r,l2)
            if (method == 'QOFW'):
                if (l2%n == 0):
                    np.random.seed(2111983 + l2/n)
                    permu = np.random.permutation(range(n))
                index = permu[l2%n]
                X_l = X.take(index, axis = 0) 
                y_l = y.take(index, axis = 0)
                res = IterationOFW(X_l,y_l,thet,bet,gam,p,Q,1,m,r,l2)
            thet = res[0]
            bet = res[1]
            gam = res[2]
            
            if drawObj:
                lossFunction = lossFunction(n, p, X, bet, gam, y, lamb, mu)
                GraphlossFunction.append(lossFunction)
            if drawPartial:
                print "bet = " , bet 
                print "gradbet(X,y,n,p,bet,gam) = ", gradbet(X,y,n,p,bet,gam,0)
                for i in range(len(gam)):
                    draw1DGam(n, p, X, bet, gam, m, y, lamb, mu, i, 100)
                    print "i = ", i, "gam = ", gam  
                for i in range(len(bet)):
                    draw1DBet(n, p, X, bet, gam, y, lamb, mu, i, 100)
            
    #print "Like = ", like(n,p,X,bet,gam,y) , "l = ", l2 
                
    # Fin de l'alogorithme de d'optimisation. Mise en forme des outputs.
    Support=(abs(bet)>.01)
    if (lamb == 0 and (method == 'N' or method == 'GS' or method == 'M')):  # Si lamba = 0, on conserve tout le support (mais pourquoi déjà????)
        Support = (abs(bet)> -1)
    supp=np.zeros((np.sum(Support)))
    SizeS=size(supp, axis=0)  # SizeS est la nouvelle valeur de p.
    if SizeS > 70:
        worked = 0 # Ca signifie qu'il y a une erreur je suppose.
    else: 
        worked = 1
    save('bet',bet)
    jj=0
    for j in range(p):
        if Support[j]:
            supp[jj]=j
            jj=jj+1
        
        if math.isnan(bet[j,0]):
            if j == 0:
                print "erreur, le ", j, "eme coefficient de beta est ", bet[j,0]
            worked = 0
            
    if SizeS > 0:            
        bet=np.transpose(np.asmatrix(bet.take([supp])))                                       
    if SizeS == 0:
        bet = []
    if drawObj:
        GraphlossFunction = np.vstack(GraphlossFunction) # Ce paragraphe sert à tracer le graph de la fonction objective.
        plt.plot(GraphlossFunction)
        if frein:
            chaine1 = 'avec recherche lineaire, '
            chaine2 = '_avec_recherche_lineaire'
        if (not frein):
            chaine1 = ''
            chaine2 = ''
        if (method == 'Nesterov'):
            plt.title('function Objective, Nesterov '+ chaine1 +  'lambda = '  + str(lamb))
            plt.savefig('function_Objective,_Nesterov' + chaine2 + '_lambda_=_' + str(lamb))
        if (method == 'GradianSimple'):
            plt.title('function Objective, gradian simple ' + chaine1)
            plt.savefig('function_Objective,_gradian_simple' + chaine2)
        if (method == 'FW'):
            plt.title('function Objective, FW ')
            plt.savefig('function_Objective, FW')
        if (method == 'OFW'):
            plt.title('function Objective, Online FW ')
            plt.savefig('function_Objective, Online FW')
        plt.show()
        
    return [bet,gam,supp,worked,h] # Ajouter FinalLogLike ou l'état final de la function objective. 


#######################################################################################################################################################
# III) fonctions servant aux verifications
#######################################################################################################################################################



def like(n, p, X, bet, gam, y):
    '''
    Cette fonction caclule le log-likeliwood des paramètre  sachant X et y.
    C'est une sous-function de "lossFunction"
    '''
    Xbet = Xbet=np.dot(X,bet)
    if bet == []:
        Xbet = np.zeros((n,1))
    cc=np.vstack([-float('inf'),gam,float('inf')])
    like=0
    for i in range(n):
        qi=int(y[i])
        like += log(ph(Xbet[i,0]-cc[qi-1,0]) - ph(Xbet[i,0]-cc[qi,0]))
    return like


def lossFunction(n, p, X, bet, gam, y, lamb, mu):
    lonemu=np.zeros((p,1))
    for j in range(p):
        if abs(bet[j,0]) < mu:
            lonemu[j,0] = bet[j,0] ** 2 / (2. * mu)
        else:
            lonemu[j,0] = abs(bet[j,0]) - mu / 2.
    return like(n,p,X,bet,gam,y) - lamb * sum(lonemu)


def draw1DGam(n, p, X, bet, gam, m, y, lamb, mu, i, N):
    '''
    i = L'indice de gamma qu'on fait bouger. Selon la valeur de i, on fait bouger gamma_1 ou gamma_2 etc.    N = Le nombre de points considérés dans le balayage.
    Tout les autre paramètres sont sont ceux de lossFunction.
    '''
    abscissa = []
    ordinate = []
    gamLocal = gam + 0.
    gam0 = gamLocal[i,0] + 0.
    grad = gradgam(X,y,n,p,m,bet,gamLocal)[i]
    if i == 0:
        for k in range(N):
            gamLocal[0,0] = gam0 - 1 + float(k)/(N-1)
            abscissa.append(gamLocal[0,0])
            ordinate.append(lossFunction(n, p, X, bet, gamLocal, y, lamb, mu))
    if i >= 1 and i < (len(gamLocal)-1):
        #print "coucou, i = ", i
        #print "gamLocal = ", gamLocal
        gamMoins1 = gamLocal[(i-1),0] + 0.
        gam1 = gamLocal[(i+1),0] + 0.
        #print "gamMoins1 = ", gamMoins1
        for k in range(N):
            gamLocal[i,0] = gamMoins1 + 10**(-4) + float(gam1 - gamMoins1 - 2 * 10**(-4))*float(k) / (N-1.)
            abscissa.append(gamLocal[i,0])
            ordinate.append(lossFunction(n, p, X, bet, gamLocal, y, lamb, mu))
    if i == (len(gamLocal)-1):
        for k in range(N):
            gamLocal[i,0] = gam0 + float(k) / (N-1.)
            abscissa.append(gamLocal[i,0])
            ordinate.append(lossFunction(n, p, X, bet, gamLocal, y, lamb, mu))

    #if i == 2 :
    #    print "abscissa = ", abscissa
    #    print "ordinate = ", ordinate
    Min = min(ordinate)
    Max = max(ordinate)
    plt.plot(abscissa, ordinate)
    plt.plot([gam0,gam0],[Min,Max])
    plt.title('Gamma_'+str(i)+"="+str(gam0)+"  Grad = " + str(grad)) # + "Min = "+ str(Min) + "Max = " + str(Max))
    plt.show()


def draw1DBet(n, p, X, bet, gam, y, lamb, mu, i, N):
    '''
    i = L'indice de beta qu'on fait bouger. Selon la valeur de i, on fait bouger Bet_1 ou Bet_2 etc.
    N = Le nombre de points considérés dans le balayage.
    Tout les autre paramètres sont sont ceux de lossFunction.
    '''
    abscissa = []
    ordinate = []
    betLocal = bet + 0. # Variable locale contenant le vecteur bet.
    bet0 = betLocal[i,0] + 0.
    grad = gradbetPenalise(X,y,n,p,betLocal,gam,lamb,mu)[i]
    for k in range(N):
        betLocal[i,0] = bet0 - 1 + (2.*k) / (N-1.)
        abscissa.append(betLocal[i,0])
        ordinate.append(lossFunction(n, p, X, betLocal, gam, y, lamb, mu))

    Min = min(ordinate)
    Max = max(ordinate)
    plt.plot(abscissa, ordinate)
    plt.plot([bet0, bet0],[Min,Max])
    plt.title('Beta_'+str(i)+"="+ str(bet0) + "  Grad = "+ str(grad))
    plt.show()

