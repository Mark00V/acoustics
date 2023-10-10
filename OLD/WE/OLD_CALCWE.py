# Benötigte Module
import numpy as np
from numpy import random
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.spatial import distance

# --------------------------------------------------
# Berechnungsparameter
print("==================================================================")
print("Berechnungsparameter")
freq = float(input("Frequenz in Hz: "))
pos_quelle_x = float(input("X-Position der Quelle in m: "))
pos_quelle_y = float(input("Y-Position der Quelle in m: "))
pos_quelle = (pos_quelle_x, pos_quelle_y)
Zi = float(input("Impedanz der Begrenzungen in kg/(m²s): "))

imp_lq = input("Schallharte Begrenzung links y/n? ")
if imp_lq == "y":
    imp_lq = False
elif imp_lq == "n":
    imp_lq = True
else:
    imp_lq = True
    print("Eingabefehler! Schallharte Begrenzung links gesetzt!")

imp_oq = input("Schallharte Begrenzung oben y/n? ")
if imp_oq == "y":
    imp_oq = False
elif imp_oq == "n":
    imp_oq = True
else:
    imp_oq = True
    print("Eingabefehler! Schallharte Begrenzung oben gesetzt!")

imp_rq = input("Schallharte Begrenzung rechts y/n? ")
if imp_rq == "y":
    imp_rq = False
elif imp_rq == "n":
    imp_rq = True
else:
    imp_rq = True
    print("Eingabefehler! Schallharte Begrenzung rechts gesetzt!")
impboundarieslor = [imp_lq, imp_oq, imp_rq]  # ImpedanzRB bei: [links,oben,rechts]

cropvaluemin = float(input("Untere Grenze Darstellung Schalldruckfeld in dB: "))
cropvaluemax = float(input("Obere Grenze Darstellung Schalldruckfeld in dB: "))

c = 340
rohair = 1.21

imp_rbs_print = []
if impboundarieslor[0] == True:
    imp_rbs_print.append("links")
if impboundarieslor[1] == True:
    imp_rbs_print.append("oben")
if impboundarieslor[2] == True:
    imp_rbs_print.append("rechts")

print("------------------------------------------------------------------")
print("Folgende Berechnungsparameter wurden gesetzt:")
print("Frequenz: ", freq)
print("Schallgeschwindigkeit: ", c)
print("Dichte Luft: ", rohair)
print("Quellenposition: ", pos_quelle)
print("Impedanz: ", Zi)
print("Impedanz Randbedingung bei (Übrige schallhart): ", imp_rbs_print)
print("Begrenzung Schalldruckfeld: ", cropvaluemin, cropvaluemax)

elementindex_data = "elementindex.csv"
pointcoordinates_data = "pointcoordinates.csv"
# --------------------------------------------------
# Meshdaten
elementindex = np.loadtxt(elementindex_data, delimiter=" ", dtype=str)
pointcoordinates = np.loadtxt(pointcoordinates_data, delimiter=" ", dtype=str)
# --------------------------------------------------
# Umrechnung Berechnungsparameter
impwert = 1 / Zi
omega = freq * 2 * math.pi
cair = c
# --------------------------------------------------

# Notwendige Funktionen und Definitionen für Berechnung Elementmatrizen p=1
# --------------------------------------------------
# Numerische Integration
numintgld1 = np.array([[0.21132486, 0.78867513], [0.50000000, 0.50000000]],
                      dtype=np.single)  # Numerische Integration 1D Knoten und Gewichte
numintgld2nodes = np.array([[0.21132487, 0.21132487], [0.21132487, 0.78867513], \
                            [0.78867513, 0.21132487], [0.78867513, 0.78867513]],
                           dtype=np.single)  # Numerische Integration 2D Knoten
numintgld2weights = np.array([0.25000000, 0.25000000, 0.25000000, 0.25000000],
                             dtype=np.single)  # Numerische Integration 2D Gewichte


# --------------------------------------------------
# Determinante der Jacobimatrix für die Eckpunkte {{x1, y1}, {x2, y2}, {x3, y3}, {x4,y4}}
# -> {{ul}, {ur}, {ol}, {or}} mit den Funktionswerten xi1 und xi2
def jacdetc(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4):
    f = 0.0 - 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
        1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - \
        1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + 1.0 * x2 * y3 + \
        1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - \
        1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4
    return f


# --------------------------------------------------
# Formfunktionen philequdist und phiqequdist
def philequdist(knoten, xi):
    knoten = int(knoten)
    if knoten == 1:
        f = 1.0 - xi
    elif knoten == 2:
        f = xi
    return f


def phiqequdist(knoten, xi1, xi2):
    knoten = int(knoten)
    if knoten == 1:
        f = (1 - xi1) * (1 - xi2)
    elif knoten == 2:
        f = xi1 * (1 - xi2)
    elif knoten == 3:
        f = (1 - xi1) * xi2
    elif knoten == 4:
        f = xi1 * xi2
    return f


def phiqequdistarray(xi1, xi2):
    f = np.array([[(1 - xi1) * (1 - xi2)], [xi1 * (1 - xi2)], [(1 - xi1) * xi2], [xi1 * xi2]], dtype=np.single)
    return f


# --------------------------------------------------
# phiqequdistgrad
def phiqequdistgrad(knoten, xi1, xi2):
    knoten = int(knoten)
    if knoten == 1:
        f = np.array([-1 + xi2, -1 + xi1], dtype=np.single)
    elif knoten == 2:
        f = np.array([1 - xi2, -xi1], dtype=np.single)
    elif knoten == 3:
        f = np.array([-xi2, 1 - xi1], dtype=np.single)
    elif knoten == 4:
        f = np.array([xi2, xi1], dtype=np.single)
    return f


# --------------------------------------------------
# gradmatequdist
def gradmatequdist(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4):
    jacinvtrans = np.array(
        [[(-1.0 * (1.0 - 1.0 * xi1) * y1 - 1.0 * xi1 * y2 + (1.0 - 1.0 * xi1) * y3 + xi1 * y4) / (0.0 - \
                                                                                                  1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
                                                                                                  1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + \
                                                                                                  1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - \
                                                                                                  1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - \
                                                                                                  1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + \
                                                                                                  1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4), \
          (1.0 * (1.0 - 1.0 * xi2) * y1 - (1.0 - 1.0 * xi2) * y2 + \
           1.0 * xi2 * y3 - xi2 * y4) / (0.0 - 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + \
                                         1.0 * x4 * xi1 * y1 + 1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - \
                                         1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + \
                                         1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - \
                                         1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + \
                                         1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4)], \
         [(1.0 * x1 * (1.0 - 1.0 * xi1) - x3 * (1.0 - 1.0 * xi1) + 1.0 * x2 * xi1 - x4 * xi1) / (0.0 - \
                                                                                                 1.0 * x2 * y1 + 1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + \
                                                                                                 1.0 * x2 * xi2 * y1 - 1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + \
                                                                                                 1.0 * x3 * xi1 * y2 - 1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - \
                                                                                                 1.0 * x1 * y3 + 1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - \
                                                                                                 1.0 * x2 * xi2 * y3 + 1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + \
                                                                                                 1.0 * x1 * xi2 * y4 - 1.0 * x3 * xi2 * y4), \
          (-1.0 * x1 * (1.0 - 1.0 * xi2) + \
           x2 * (1.0 - 1.0 * xi2) - 1.0 * x3 * xi2 + x4 * xi2) / (0.0 - 1.0 * x2 * y1 + \
                                                                  1.0 * x3 * y1 - 1.0 * x3 * xi1 * y1 + 1.0 * x4 * xi1 * y1 + 1.0 * x2 * xi2 * y1 - \
                                                                  1.0 * x4 * xi2 * y1 + 1.0 * x1 * y2 - 1.0 * x3 * y2 + 1.0 * x3 * xi1 * y2 - \
                                                                  1.0 * x4 * xi1 * y2 - 1.0 * x1 * xi2 * y2 + 1.0 * x3 * xi2 * y2 - 1.0 * x1 * y3 + \
                                                                  1.0 * x2 * y3 + 1.0 * x1 * xi1 * y3 - 1.0 * x2 * xi1 * y3 - 1.0 * x2 * xi2 * y3 + \
                                                                  1.0 * x4 * xi2 * y3 - 1.0 * x1 * xi1 * y4 + 1.0 * x2 * xi1 * y4 + 1.0 * x1 * xi2 * y4 - \
                                                                  1.0 * x3 * xi2 * y4)]], dtype=np.single)

    vekgradmat = np.zeros((4, 2), dtype=np.single)

    for knoten in range(4):
        matrow = jacinvtrans @ phiqequdistgrad(knoten + 1, xi1, xi2)
        vekgradmat[knoten] = matrow
    return vekgradmat


# --------------------------------------------------

# --------------------------------------------------
# Import Meshdata
elementindex = elementindex[:, 0:4]
elementindex = elementindex.astype(np.uint32)

pointcoordinates = pointcoordinates[:, 0:2]
pointcoordinates = pointcoordinates.astype(np.float32)

maxelement = len(elementindex)
maxnode = len(pointcoordinates)
zuordtab = elementindex
allnodes = np.zeros((maxnode, 3))

k = 1
for i in range(maxnode):
    allnodes[i, 0] = pointcoordinates[i, 0]
    allnodes[i, 1] = pointcoordinates[i, 1]
    allnodes[i, 2] = int(k)
    k += 1
print("------------------------------------------------------------------")
print("Informationen Netz:")
print("Anzahl Elemente: " + str(maxelement))
print("Anzahl Freiheitsgrade: " + str(maxnode))
print("------------------------------------------------------------------")
# --------------------------------------------------
# Boundaries
minxpc = min(allnodes[:, 0])
maxxpc = max(allnodes[:, 0])
minypc = min(allnodes[:, 1])
maxypc = max(allnodes[:, 1])

boundaryleft = allnodes[allnodes[:, 0] == minxpc]
boundaryright = allnodes[allnodes[:, 0] == maxxpc]
boundarybottom = allnodes[allnodes[:, 1] == minypc]
boundarytop = allnodes[allnodes[:, 1] == maxypc]

# --------------------------------------------------
# Randbedingungen Impedanz, Beachte Indexierung hier gegenüber struct mesh (allnods[:,2] startet hier mit 1!
boundaryleftn = boundaryleft[:, 2] - 1
boundaryrightn = boundaryright[:, 2] - 1
boundarybottomn = boundarybottom[:, 2] - 1
boundarytopn = boundarytop[:, 2] - 1

zuordtableft_ibc = [boundaryleftn[i:i + 2] for i in range(0, len(boundaryleftn) - 1, 1)]
zuordtabright_ibc = [boundaryrightn[i:i + 2] for i in range(0, len(boundaryrightn) - 1, 1)]
zuordtablebottom_ibc = [boundarybottomn[i:i + 2] for i in range(0, len(boundarybottomn) - 1, 1)]
zuordtabtop_ibc = [boundarytopn[i:i + 2] for i in range(0, len(boundarytopn) - 1, 1)]

# --------------------------------------------------
# Erstelle Elementmatrizen
startTime = time.time()
print("Erstellung Elementmatrizen")

intnodes = numintgld2nodes
intweights = numintgld2weights
allelementssteif = np.zeros((maxelement, 4, 4), dtype=np.single)
allelementsmass = np.zeros((maxelement, 4, 4), dtype=np.single)

for ie in range(maxelement):
    elesteifmat = np.zeros((4, 4), dtype=np.single)
    elemassenmat = np.zeros((4, 4), dtype=np.single)

    ielem = zuordtab[ie]
    knoten1 = ielem[0]
    knoten2 = ielem[1]
    knoten3 = ielem[2]
    knoten4 = ielem[3]
    x1 = pointcoordinates[knoten1, 0]
    y1 = pointcoordinates[knoten1, 1]
    x2 = pointcoordinates[knoten2, 0]
    y2 = pointcoordinates[knoten2, 1]
    x3 = pointcoordinates[knoten3, 0]
    y3 = pointcoordinates[knoten3, 1]
    x4 = pointcoordinates[knoten4, 0]
    y4 = pointcoordinates[knoten4, 1]

    for i in range(4):
        xi1 = intnodes[i, 0]
        xi2 = intnodes[i, 1]
        gr = gradmatequdist(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4)
        grt = np.transpose(gr)
        grxgrt = gr @ grt
        jd = jacdetc(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4)
        fp = grxgrt * jd * intweights
        elesteifmat = elesteifmat + fp

    for i in range(4):
        xi1 = intnodes[i, 0]
        xi2 = intnodes[i, 1]
        phi = phiqequdistarray(xi1, xi2)
        phit = np.transpose(phiqequdistarray(xi1, xi2))
        phixphit = phi @ phit
        jd = jacdetc(xi1, xi2, x1, x2, x3, x4, y1, y2, y3, y4)
        fp = phixphit * jd * intweights
        elemassenmat = elemassenmat + fp

    allelementssteif[ie] = elesteifmat
    allelementsmass[ie] = elemassenmat

# --------------------------------------------------
# Erstelle Randelementematrizen Impedanz

intnodes = numintgld1[0]
intweights = numintgld1[1]

# links
neumannwert = impwert
allimpelementslinks = np.zeros((len(zuordtableft_ibc), 2, 2))
for ielem in range(0, len(zuordtableft_ibc)):
    impmatele = np.zeros((2, 2))
    nb = np.zeros((2, 2))
    element = zuordtableft_ibc[ielem]
    knoten1 = int(element[0])
    knoten2 = int(element[1])
    pos_y_knoten1 = allnodes[knoten1, 1]
    pos_y_knoten2 = allnodes[knoten2, 1]
    length = abs(pos_y_knoten2 - pos_y_knoten1)
    for j in range(0, 2):
        for i in range(0, 2):
            for ii in range(0, 2):
                val = philequdist(i + 1, intnodes[j]) * philequdist(ii + 1, intnodes[j]) * intweights[j]
                nb[i, ii] = val
        impmatele = impmatele + nb
    impmatele = impmatele * length * neumannwert
    allimpelementslinks[ielem] = impmatele

# oben
neumannwert = impwert
allimpelementsoben = np.zeros((len(zuordtabtop_ibc), 2, 2))
for ielem in range(0, len(zuordtabtop_ibc)):
    impmatele = np.zeros((2, 2))
    nb = np.zeros((2, 2))
    element = zuordtabtop_ibc[ielem]
    knoten1 = int(element[0])
    knoten2 = int(element[1])
    pos_x_knoten1 = allnodes[knoten1, 0]
    pos_x_knoten2 = allnodes[knoten2, 0]
    length = abs(pos_y_knoten2 - pos_y_knoten1)
    for j in range(0, 2):
        for i in range(0, 2):
            for ii in range(0, 2):
                val = philequdist(i + 1, intnodes[j]) * philequdist(ii + 1, intnodes[j]) * intweights[j]
                nb[i, ii] = val
        impmatele = impmatele + nb
    impmatele = impmatele * length * neumannwert
    allimpelementsoben[ielem] = impmatele

# rechts
neumannwert = impwert
allimpelementsrechts = np.zeros((len(zuordtabright_ibc), 2, 2))
for ielem in range(0, len(zuordtabright_ibc)):
    impmatele = np.zeros((2, 2))
    nb = np.zeros((2, 2))
    element = zuordtabright_ibc[ielem]
    knoten1 = int(element[0])
    knoten2 = int(element[1])
    pos_y_knoten1 = allnodes[knoten1, 1]
    pos_y_knoten2 = allnodes[knoten2, 1]
    length = abs(pos_y_knoten2 - pos_y_knoten1)
    for j in range(0, 2):
        for i in range(0, 2):
            for ii in range(0, 2):
                val = philequdist(i + 1, intnodes[j]) * philequdist(ii + 1, intnodes[j]) * intweights[j]
                nb[i, ii] = val
        impmatele = impmatele + nb
    impmatele = impmatele * length * neumannwert
    allimpelementsrechts[ielem] = impmatele

executionTime = (time.time() - startTime)
print('\nErstellung Elementmatrizen beendet: ' + str(executionTime) + "s")
# --------------------------------------------------
# Erstelle Impedanzmatrix

# links
sysimparraylinks = np.zeros((maxnode, maxnode))
for ielem in range(len(zuordtableft_ibc)):
    eleimpmat = allimpelementslinks[ielem]
    for a in range(2):
        for b in range(2):
            zhab = zuordtableft_ibc[ielem]
            zta = int(zhab[a])
            ztb = int(zhab[b])
            sysimparraylinks[zta, ztb] = sysimparraylinks[zta, ztb] + eleimpmat[a, b]

# oben
sysimparrayoben = np.zeros((maxnode, maxnode))
for ielem in range(len(zuordtabtop_ibc)):
    eleimpmat = allimpelementsoben[ielem]
    for a in range(2):
        for b in range(2):
            zhab = zuordtabtop_ibc[ielem]
            zta = int(zhab[a])
            ztb = int(zhab[b])
            sysimparrayoben[zta, ztb] = sysimparrayoben[zta, ztb] + eleimpmat[a, b]

# rechts
sysimparrayrechts = np.zeros((maxnode, maxnode))
for ielem in range(len(zuordtabright_ibc)):
    eleimpmat = allimpelementsrechts[ielem]
    for a in range(2):
        for b in range(2):
            zhab = zuordtabright_ibc[ielem]
            zta = int(zhab[a])
            ztb = int(zhab[b])
            sysimparrayrechts[zta, ztb] = sysimparrayrechts[zta, ztb] + eleimpmat[a, b]

sysimparray = np.zeros((maxnode, maxnode))

if impboundarieslor[0] == True:
    sysimparray = sysimparray + sysimparraylinks
if impboundarieslor[1] == True:
    sysimparray = sysimparray + sysimparrayoben
if impboundarieslor[2] == True:
    sysimparray = sysimparray + sysimparrayrechts

# ---------  Assemblierung
startTime = time.time()
print("Assemblierung Systemmatrizen")

elesteifmatarray = [allelementssteif[i] for i in range(maxelement)]
elesteifmatarray = np.asarray(elesteifmatarray)

elesmassmatarray = [allelementsmass[i] for i in range(maxelement)]
elesmassmatarray = np.asarray(elesmassmatarray)

vectora = [[[int(zuordtab[ielem, a]) for b in range(4)] for a in range(4)] for ielem in range(maxelement)]
vectora = np.asarray(vectora)
vectora = vectora.flatten()
row = vectora

vectorb = [[[int(zuordtab[ielem, b]) for b in range(4)] for a in range(4)] for ielem in range(maxelement)]
vectorb = np.asarray(vectorb)
vectorb = vectorb.flatten()
col = vectorb

# Steifigkeitsmatrix
data = elesteifmatarray.flatten()
syssteifarraycoo = coo_matrix((data, (row, col)), shape=(maxnode, maxnode))

# Massenmatrix
data = elesmassmatarray.flatten()
sysmassarraycoo = coo_matrix((data, (row, col)), shape=(maxnode, maxnode))

syssteif = syssteifarraycoo
sysmass = sysmassarraycoo

sysmatfreq = (1 / rohair) * syssteif - (1 / rohair) * (1 / cair ** 2) * (
            omega ** 2) * sysmass + omega * sysimparray * 1j
sysmatfreqh = (1 / rohair) * syssteif - (1 / rohair) * (1 / cair ** 2) * (
            omega ** 2) * sysmass + omega * sysimparray * 1j  # WARUM IST DAS NOTWENDIG???? Wieso werden die Werte nicht einfach neu zugewiesen?

executionTime = (time.time() - startTime)
print("Assemblierung Systemmatrizen beendet: " + str(executionTime) + "s")


# ---------
# --------------------------------------------------
# Quelle Monopolquelle und Einbau in Lastvektor
# Position Monopolquelle hier x=0.1,y=0.2
def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


cn_posquelle = closest_node(pos_quelle, allnodes[:, 0:2])
cn_allnodes = allnodes[(allnodes[:, 0] == cn_posquelle[0]) & (allnodes[:, 1] == cn_posquelle[1])]

# Einbau Monopolquelle
posknoten = int(cn_allnodes[0, 2])
mpsource = 4 * math.pi / rohair * ((2 * rohair * omega) / ((2 * math.pi) ** 2)) ** 0.5
source = np.array([posknoten, mpsource])

lastvektor = np.zeros(maxnode)
lastvektor[int(source[0])] = lastvektor[int(source[0])] + source[1]

# --------------------------------------------------
# Einarbeitung übriger Randbedingungen
# Hier nicht nötig, da Neumann
sysmatfreqred = sysmatfreq

# --------------------------------------------------
# Lösung GLS
startTime = time.time()
print("Berechne Lösungsvektor")

lastvektorrows = np.reshape(lastvektor, (maxnode, 1))

sysmatfreqredcsc = csc_matrix(sysmatfreqred, dtype=complex)
lastvektorcsc = csc_matrix(lastvektorrows, dtype=complex)
solution = spsolve(sysmatfreqredcsc, lastvektorcsc)
solutionallnodes = np.zeros((maxnode, 3))
for i in range(maxnode):
    solutionallnodes[i, 0] = allnodes[i, 0]
    solutionallnodes[i, 1] = allnodes[i, 1]
    solutionallnodes[i, 2] = np.real(solution[i])
# ----
executionTime = (time.time() - startTime)
print("Berechne Lösungsvektor beendet: " + str(executionTime) + "s")
# --------------------------------------------------
# Berechnung Schalldrucklevel
pref = 20 * 10 ** (-6)
solutiondruck = solution  # Druck (mit Im-Teil)
solutionspl = abs(20 * np.log10(solution / pref))  # Schalldrucklevel


# --------------------------------------------------
# Crop data
def datacropped(data, minvalue, maxvalue):
    lendata = len(data)
    newdata = np.zeros(lendata)
    for i in range(lendata):
        newdata[i] = data[i]
        if data[i] < minvalue:
            newdata[i] = minvalue
        if data[i] > maxvalue:
            newdata[i] = maxvalue

    return newdata


newspldata = datacropped(solutionspl, cropvaluemin, cropvaluemax)

# Ausgabe Contourplot
print("Berechne grafische Ausgabe")
print("==================================================================")
datax = solutionallnodes[:, 0]
datay = solutionallnodes[:, 1]
dataz = np.real(newspldata)

minx = min(datax)
maxx = max(datax)
miny = min(datay)
maxy = max(datay)

points = solutionallnodes[:, (0, 1)]
values = dataz

grid_x, grid_y = np.mgrid[minx:maxx:600j, miny:maxy:600j]

grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

# Contourplot
nr_of_contours = 100  # Contouren insgesamt, hoher Wert für Quasi-Densityplot
nr_of_contourlines = 5  # EIngezeichnete Contourlinien, Wert nicht exakt...
aspectxy = 1
ctlines = int(nr_of_contours / nr_of_contourlines)

dataX = grid_x
dataY = grid_y
dataZ = grid_z1

fig1, ax = plt.subplots()
CS1 = ax.contourf(dataX, dataY, dataZ, nr_of_contours, cmap=plt.cm.gnuplot2)
ax.set_title('Schalldruckfeld')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect(aspectxy)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = fig1.colorbar(CS1, cax=cax)
cbar.ax.set_ylabel('SPL [dB]')

plt.show()
